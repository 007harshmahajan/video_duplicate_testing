use opencv::{
    core::{Mat, MatTraitConst, Size},
    prelude::*,
    videoio::{VideoCapture, CAP_ANY},
    imgproc,
    Result as OpenCVResult,
};
use std::time::Instant;
use faiss::{Index, MetricType, Idx};
use uuid::Uuid;
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
struct VideoFingerprint {
    id: String,
    dhash_vectors: Vec<f32>,     // For FAISS vector search
    binary_hashes: Vec<u64>,     // For Hamming distance search
    scene_timestamps: Vec<f64>,
}

#[derive(Debug)]
struct VectorMatchResult {
    video_id: i64,
    uuid: String,
    similarity: f64,
    matching_scenes: Vec<(usize, usize, f64)>, // (search_scene_idx, db_scene_idx, similarity)
}

#[derive(Debug)]
struct HammingMatchResult {
    video_id: i64,
    uuid: String,
    similarity: f64,
    matching_scenes: Vec<(usize, usize, f64)>, // (search_scene_idx, db_scene_idx, similarity)
}

fn main() -> OpenCVResult<()> {
    let start_time = Instant::now();
    
    let dimension = 64;
    let mut index = faiss::index_factory(
        dimension,
        "IDMap,Flat",
        MetricType::L2
    ).expect("Failed to create FAISS index");

    let mut video_database: HashMap<i64, VideoFingerprint> = HashMap::new();

    let video_path = "/home/harshmahajan/Downloads/yral_ori.mp4";
    let fingerprint = process_video(video_path)?;
    add_video_to_database(&mut index, &mut video_database, fingerprint);

    let search_video_path = "/home/harshmahajan/yral1.mp4";
    let search_fingerprint = process_video(search_video_path)?;
    search_similar_videos(&mut index, &video_database, &search_fingerprint);

    let duration = start_time.elapsed();
    println!("Total processing completed in: {:.2?}", duration);

    Ok(())
}

fn calculate_hamming_distance(hash1: u64, hash2: u64) -> u32 {
    (hash1 ^ hash2).count_ones()
}

fn binary_hash_to_u64(binary: &[u8]) -> u64 {
    let mut hash: u64 = 0;
    for (i, &pixel) in binary.iter().enumerate() {
        if pixel > 127 {
            hash |= 1 << i;
        }
    }
    hash
}

fn calculate_frame_difference(prev_frame: &Mat, curr_frame: &Mat) -> OpenCVResult<(f64, Vec<f32>, u64)> {
    let mut gray_prev = Mat::default();
    let mut gray_curr = Mat::default();
    
    imgproc::cvt_color(prev_frame, &mut gray_prev, imgproc::COLOR_BGR2GRAY, 0)?;
    imgproc::cvt_color(curr_frame, &mut gray_curr, imgproc::COLOR_BGR2GRAY, 0)?;

    let target_size = Size::new(8, 8);
    let mut small_prev = Mat::default();
    let mut small_curr = Mat::default();
    imgproc::resize(&gray_prev, &mut small_prev, target_size, 0.0, 0.0, imgproc::INTER_AREA)?;
    imgproc::resize(&gray_curr, &mut small_curr, target_size, 0.0, 0.0, imgproc::INTER_AREA)?;

    let mut binary_prev = Mat::default();
    let mut binary_curr = Mat::default();
    imgproc::threshold(
        &small_prev,
        &mut binary_prev,
        127.0,
        255.0,
        imgproc::THRESH_BINARY,
    )?;
    imgproc::threshold(
        &small_curr,
        &mut binary_curr,
        127.0,
        255.0,
        imgproc::THRESH_BINARY,
    )?;

    let mut diff = Mat::default();
    opencv::core::bitwise_xor(&binary_prev, &binary_curr, &mut diff, &Mat::default())?;
    let non_zero = opencv::core::count_non_zero(&diff)?;
    let total_pixels = 64;
    let hamming_distance = non_zero as f64 / total_pixels as f64;

    let mut dhash_vector = Vec::with_capacity(64);
    let mut binary_pixels = Vec::with_capacity(64);
    
    for row in 0..8 {
        for col in 0..8 {
            let pixel = binary_curr.at_2d::<u8>(row, col)?;
            dhash_vector.push(*pixel as f32 / 255.0);
            binary_pixels.push(*pixel);
        }
    }

    let binary_hash = binary_hash_to_u64(&binary_pixels);

    Ok((hamming_distance, dhash_vector, binary_hash))
}

fn process_video(video_path: &str) -> OpenCVResult<VideoFingerprint> {
    let mut cap = VideoCapture::from_file(video_path, CAP_ANY)?;
    let fps = cap.get(opencv::videoio::CAP_PROP_FPS)?;
    let total_frames = cap.get(opencv::videoio::CAP_PROP_FRAME_COUNT)? as i32;
    
    println!("Processing video: {}", video_path);
    println!("FPS: {}, Total frames: {}", fps, total_frames);
    
    let mut dhash_vectors = Vec::new();
    let mut binary_hashes = Vec::new();
    let mut scene_timestamps = Vec::new();
    let mut prev_frame = Mat::default();
    let threshold = 0.2;
    let mut frame_number = 0;
    let mut scenes_detected = 0;

    loop {
        let mut frame = Mat::default();
        if !cap.read(&mut frame)? {
            break;
        }
        frame_number += 1;

        if !prev_frame.empty() {
            let (hamming_dist, dhash, binary_hash) = calculate_frame_difference(&prev_frame, &frame)?;
            
            if hamming_dist > threshold {
                dhash_vectors.extend(dhash);
                binary_hashes.push(binary_hash);
                scene_timestamps.push(frame_number as f64 / fps);
                scenes_detected += 1;

                if scenes_detected % 10 == 0 {
                    println!("Detected {} scenes...", scenes_detected);
                }
            }
        }

        frame.copy_to(&mut prev_frame)?;
    }

    println!("Video processing complete:");
    println!("Total frames processed: {}", frame_number);
    println!("Total scenes detected: {}", scenes_detected);

    if dhash_vectors.is_empty() {
        println!("No scenes detected, storing initial frame...");
        let (_, dhash, binary_hash) = calculate_frame_difference(&prev_frame, &prev_frame)?;
        dhash_vectors.extend(dhash);
        binary_hashes.push(binary_hash);
        scene_timestamps.push(0.0);
    }

    Ok(VideoFingerprint {
        id: Uuid::new_v4().to_string(),
        dhash_vectors,
        binary_hashes,
        scene_timestamps,
    })
}

fn add_video_to_database(
    index: &mut impl Index,
    database: &mut HashMap<i64, VideoFingerprint>,
    fingerprint: VideoFingerprint
) {
    let base_id = database.len() as i64;
    
    for (i, chunk) in fingerprint.dhash_vectors.chunks_exact(64).enumerate() {
        let id = base_id + i as i64;
        let idx: Idx = id.into();
        index.add_with_ids(
            &chunk.to_vec(),
            &[idx]
        ).expect("Failed to add vector to index");
    }
    
    database.insert(base_id, fingerprint);
    println!("Added video to database with ID: {}", base_id);
}

fn search_similar_videos(
    index: &mut impl Index,
    database: &HashMap<i64, VideoFingerprint>,
    search_fingerprint: &VideoFingerprint
) {
    let k = 20;
    let vector_threshold = 0.4f64;
    let hamming_threshold = 0.5f64;
    
    // Track matches for both methods
    let mut vector_matches = HashMap::new();
    let mut hamming_matches = HashMap::new();

    println!("\nAnalyzing {} scenes from search video...", search_fingerprint.binary_hashes.len());

    // Vector similarity search using FAISS
    for (search_scene_idx, chunk) in search_fingerprint.dhash_vectors.chunks_exact(64).enumerate() {
        let result = index.search(
            &chunk.to_vec(),
            k
        ).expect("Failed to search index");

        for (i, label) in result.labels.iter().enumerate() {
            let distance = result.distances[i] as f64;
            let vector_similarity = (-distance/64.0).exp(); // Cosine similarity conversion

            if vector_similarity >= vector_threshold {
                if let Some(id) = label.get() {
                    let video_id = (id % database.len() as u64) as i64;
                    let db_scene_idx = (id / database.len() as u64) as usize;
                    
                    vector_matches
                        .entry(video_id)
                        .or_insert_with(VectorMatchResult::new)
                        .add_match(search_scene_idx, db_scene_idx, vector_similarity);
                }
            }
        }
    }

    // Hamming distance search using multi-index hashing
    for (search_scene_idx, search_hash) in search_fingerprint.binary_hashes.iter().enumerate() {
        // Split hash into 4 parts for multi-index search
        let hash_parts = split_hash(search_hash);
        
        for (video_id, db_fingerprint) in database {
            for (db_scene_idx, db_hash) in db_fingerprint.binary_hashes.iter().enumerate() {
                let db_hash_parts = split_hash(db_hash);
                
                // Check if any part matches exactly (pigeonhole principle)
                let mut has_matching_part = false;
                for i in 0..4 {
                    if hash_parts[i] == db_hash_parts[i] {
                        has_matching_part = true;
                        break;
                    }
                }

                if has_matching_part {
                    // Calculate full Hamming distance only for candidates
                    let hamming_similarity = 1.0 - (hamming_distance(search_hash, db_hash) as f64 / 64.0);
                    
                    if hamming_similarity >= hamming_threshold {
                        hamming_matches
                            .entry(*video_id)
                            .or_insert_with(HammingMatchResult::new)
                            .add_match(search_scene_idx, db_scene_idx, hamming_similarity);
                    }
                }
            }
        }
    }

    println!("\nProcessing Results:");
    println!("Found {} potential vector matches", vector_matches.len());
    println!("Found {} potential hamming matches", hamming_matches.len());

    // Process vector similarity results
    let mut vector_results: Vec<VectorMatchResult> = vector_matches
        .into_iter()
        .map(|(_, mut result)| {
            let scene_count = result.matching_scenes.len();
            if scene_count > 0 {
                result.similarity = result.matching_scenes.iter()
                    .map(|&(_, _, sim)| sim)
                    .sum::<f64>() / scene_count as f64;
            }
            result
        })
        .collect();

    // Process hamming distance results
    let mut hamming_results: Vec<HammingMatchResult> = hamming_matches
        .into_iter()
        .map(|(_, mut result)| {
            let scene_count = result.matching_scenes.len();
            if scene_count > 0 {
                result.similarity = result.matching_scenes.iter()
                    .map(|&(_, _, sim)| sim)
                    .sum::<f64>() / scene_count as f64;
            }
            result
        })
        .collect();

    // Sort results by similarity
    vector_results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
    hamming_results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));

    // Print Vector Similarity Results
    println!("\nVector Similarity Results:");
    println!("-------------------------");
    for result in vector_results {
        if result.similarity >= vector_threshold {
            println!("Video ID: {}, UUID: {}", result.video_id, result.uuid);
            println!("Similarity: {:.2}%", result.similarity * 100.0);
            println!("Matching Scenes: {}", result.matching_scenes.len());
            
            println!("Top 5 matching scenes (Search Scene → DB Scene):");
            let mut top_scenes = result.matching_scenes;
            top_scenes.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            
            for (search_idx, db_idx, similarity) in top_scenes.iter().take(5) {
                println!(
                    "  Scene {} → Scene {}: {:.2}% similarity",
                    search_idx,
                    db_idx,
                    similarity * 100.0
                );
            }
            println!("-------------------------");
        }
    }

    // Print Hamming Distance Results
    println!("\nHamming Distance Results:");
    println!("-------------------------");
    for result in hamming_results {
        if result.similarity >= hamming_threshold {
            println!("Video ID: {}, UUID: {}", result.video_id, result.uuid);
            println!("Similarity: {:.2}%", result.similarity * 100.0);
            println!("Matching Scenes: {}", result.matching_scenes.len());
            
            println!("Top 5 matching scenes (Search Scene → DB Scene):");
            let mut top_scenes = result.matching_scenes;
            top_scenes.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            
            for (search_idx, db_idx, similarity) in top_scenes.iter().take(5) {
                println!(
                    "  Scene {} → Scene {}: {:.2}% similarity",
                    search_idx,
                    db_idx,
                    similarity * 100.0
                );
            }
            println!("-------------------------");
        }
    }
}

// Helper functions
fn split_hash(hash: &u64) -> [u16; 4] {
    [
        ((hash >> 48) & 0xFFFF) as u16,
        ((hash >> 32) & 0xFFFF) as u16,
        ((hash >> 16) & 0xFFFF) as u16,
        (hash & 0xFFFF) as u16
    ]
}

fn hamming_distance(a: &u64, b: &u64) -> u32 {
    (a ^ b).count_ones()
}

impl VectorMatchResult {
    fn new() -> Self {
        Self {
            video_id: 0,
            uuid: String::new(),
            similarity: 0.0,
            matching_scenes: Vec::new()
        }
    }

    fn add_match(&mut self, search_idx: usize, db_idx: usize, similarity: f64) {
        self.matching_scenes.push((search_idx, db_idx, similarity));
        // Update overall similarity as average
        self.similarity = self.matching_scenes.iter()
            .map(|(_, _, sim)| sim)
            .sum::<f64>() / self.matching_scenes.len() as f64;
    }
}

impl HammingMatchResult {
    fn new() -> Self {
        Self {
            video_id: 0,
            uuid: String::new(),
            similarity: 0.0,
            matching_scenes: Vec::new()
        }
    }

    fn add_match(&mut self, search_idx: usize, db_idx: usize, similarity: f64) {
        self.matching_scenes.push((search_idx, db_idx, similarity));
        // Update overall similarity as average
        self.similarity = self.matching_scenes.iter()
            .map(|(_, _, sim)| sim)
            .sum::<f64>() / self.matching_scenes.len() as f64;
    }
}

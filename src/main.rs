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
use std::collections::HashMap;

struct VideoFingerprint {
    id: String,
    dhash_vectors: Vec<f32>,
    scene_timestamps: Vec<f64>,
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

    let video_path = "";
    let fingerprint = process_video(video_path)?;
    add_video_to_database(&mut index, &mut video_database, fingerprint);

    let search_video_path = "";
    let search_fingerprint = process_video(search_video_path)?;
    search_similar_videos(&mut index, &video_database, &search_fingerprint);

    let duration = start_time.elapsed();
    println!("Total processing completed in: {:.2?}", duration);

    Ok(())
}

fn process_video(video_path: &str) -> OpenCVResult<VideoFingerprint> {
    let mut cap = VideoCapture::from_file(video_path, CAP_ANY)?;
    let fps = cap.get(opencv::videoio::CAP_PROP_FPS)?;
    let total_frames = cap.get(opencv::videoio::CAP_PROP_FRAME_COUNT)? as i32;
    
    println!("Processing video: {}", video_path);
    println!("FPS: {}, Total frames: {}", fps, total_frames);
    
    let mut scene_dhashes = Vec::new();
    let mut scene_timestamps = Vec::new();
    let mut prev_frame = Mat::default();
    let threshold = 0.3;
    let mut frame_number = 0;
    let mut scenes_detected = 0;

    loop {
        let mut frame = Mat::default();
        if !cap.read(&mut frame)? {
            break;
        }
        frame_number += 1;

        if !prev_frame.empty() {
            let (hamming_dist, dhash) = calculate_frame_difference(&prev_frame, &frame)?;
            
            if hamming_dist > threshold {
                scene_dhashes.extend(dhash);
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
    println!("Total dhash vectors: {}", scene_dhashes.len() / 64);

    // Always store at least one frame if no scenes were detected
    if scene_dhashes.is_empty() {
        println!("No scenes detected, storing initial frame...");
        let (_, dhash) = calculate_frame_difference(&prev_frame, &prev_frame)?;
        scene_dhashes.extend(dhash);
        scene_timestamps.push(0.0);
    }

    Ok(VideoFingerprint {
        id: Uuid::new_v4().to_string(),
        dhash_vectors: scene_dhashes,
        scene_timestamps,
    })
}

fn calculate_frame_difference(prev_frame: &Mat, curr_frame: &Mat) -> OpenCVResult<(f64, Vec<f32>)> {
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
    let total_pixels = 64; // 8x8
    let hamming_distance = non_zero as f64 / total_pixels as f64;

    // Lower the threshold for scene detection
    let threshold = 0.2; // Reduced from 0.3
    let mut dhash = Vec::with_capacity(64);
    
    if hamming_distance > threshold {
        for row in 0..8 {
            for col in 0..8 {
                let pixel = binary_curr.at_2d::<u8>(row, col)?;
                dhash.push(*pixel as f32 / 255.0);
            }
        }
    } else {
        // If frames are very similar, use a more granular comparison
        for row in 0..8 {
            for col in 0..8 {
                let prev_pixel = binary_prev.at_2d::<u8>(row, col)?;
                let curr_pixel = binary_curr.at_2d::<u8>(row, col)?;
                let avg = (*prev_pixel as f32 + *curr_pixel as f32) / 510.0; // Normalize to 0-1
                dhash.push(avg);
            }
        }
    }

    Ok((hamming_distance, dhash))
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
    let k = 20; // Increased for more potential matches
    let similarity_threshold = 0.4; // Lowered threshold for more tolerance
    let min_match_percentage = 5.0; // Lowered minimum match requirement
    let mut video_similarities = HashMap::new();
    let mut scene_matches = HashMap::new();

    // Process each scene's dhash in chunks
    for (scene_idx, chunk) in search_fingerprint.dhash_vectors.chunks_exact(64).enumerate() {
        let result = index.search(
            &chunk.to_vec(),
            k
        ).expect("Failed to search index");

        // Calculate similarity for each result
        for (i, label) in result.labels.iter().enumerate() {
            let distance = result.distances[i];
            
            // Modified similarity calculation to be more tolerant
            // Using exponential decay for distance to similarity conversion
            let similarity = (-distance/64.0).exp(); // Normalized by vector dimension

            if similarity >= similarity_threshold {
                if let Some(id) = label.get() {
                    let video_id = (id % database.len() as u64) as i64;
                    
                    // Store scene-level matches with confidence
                    scene_matches
                        .entry(video_id)
                        .or_insert_with(Vec::new)
                        .push((scene_idx, similarity));

                    // Update overall video similarity with weighted scoring
                    let video_entry = video_similarities
                        .entry(video_id)
                        .or_insert((0.0, 0)); // (total_similarity, match_count)
                    
                    // Weight higher similarities more
                    let weight = similarity.powi(2);
                    video_entry.0 += weight * similarity;
                    video_entry.1 += 1;
                }
            }
        }
    }

    // Sort and display results
    let mut results: Vec<_> = video_similarities.iter().collect();
    results.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nSearch Results:");
    println!("---------------");
    
    for (video_id, (total_similarity, match_count)) in results {
        if let Some(video) = database.get(video_id) {
            let avg_similarity = total_similarity / *match_count as f32;
            let match_percentage = (*match_count as f64 / (search_fingerprint.dhash_vectors.len() / 64) as f64) * 100.0;
            
            // Show results with significant matches
            if match_percentage >= min_match_percentage {
                println!(
                    "Video ID: {}, UUID: {}",
                    video_id, video.id
                );
                println!(
                    "Match Percentage: {:.2}%, Average Similarity: {:.2}%, Matching Scenes: {}",
                    match_percentage, avg_similarity * 100.0, match_count
                );

                // Show matching scenes details with confidence levels
                if let Some(scenes) = scene_matches.get(video_id) {
                    println!("Top matching scenes (with confidence):");
                    let mut top_scenes: Vec<_> = scenes.iter().collect();
                    top_scenes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    
                    for (scene_idx, similarity) in top_scenes.iter().take(5) {
                        let confidence_level = match *similarity {
                            s if s >= 0.8 => "High",
                            s if s >= 0.6 => "Medium",
                            _ => "Low"
                        };
                        
                        println!(
                            "  Scene {}: {:.2}% similarity (Confidence: {})",
                            scene_idx,
                            similarity * 100.0,
                            confidence_level
                        );
                    }
                }
                println!("---------------");
            }
        }
    }
}

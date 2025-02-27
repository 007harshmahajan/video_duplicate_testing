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

    let video_path = "/home/harshmahajan/Downloads/yral.mp4";
    let fingerprint = process_video(video_path)?;
    add_video_to_database(&mut index, &mut video_database, fingerprint);

    let search_video_path = "/home/harshmahajan/Downloads/yral.mp4";
    let search_fingerprint = process_video(search_video_path)?;
    search_similar_videos(&mut index, &video_database, &search_fingerprint);

    let duration = start_time.elapsed();
    println!("Total processing completed in: {:.2?}", duration);

    Ok(())
}

fn process_video(video_path: &str) -> OpenCVResult<VideoFingerprint> {
    let mut cap = VideoCapture::from_file(video_path, CAP_ANY)?;
    let fps = cap.get(opencv::videoio::CAP_PROP_FPS)?;
    
    let mut scene_dhashes = Vec::new();
    let mut scene_timestamps = Vec::new();
    let mut prev_frame = Mat::default();
    let threshold = 0.3; // Adjust scene change sensitivity
    let mut frame_number = 0;

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
            }
        }

        frame.copy_to(&mut prev_frame)?;
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

    let mut dhash = Vec::with_capacity(64);
    for row in 0..8 {
        for col in 0..8 {
            let pixel = binary_curr.at_2d::<u8>(row, col)?;
            dhash.push(*pixel as f32 / 255.0);
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
    let k = 5;
    let mut match_counts = HashMap::new();

    for chunk in search_fingerprint.dhash_vectors.chunks_exact(64) {
        let result = index.search(
            &chunk.to_vec(),
            k
        ).expect("Failed to search index");

        for (i, label) in result.labels.iter().enumerate() {
            if result.distances[i] < 0.3 {
                if let Some(id) = label.get() {
                    let video_id = (id % database.len() as u64) as i64;
                    *match_counts.entry(video_id).or_insert(0) += 1;
                }
            }
        }
    }

    println!("\nSearch Results:");
    println!("---------------");
    for (video_id, matches) in match_counts.iter() {
        if let Some(video) = database.get(video_id) {
            let match_percentage = (*matches as f64 / (search_fingerprint.dhash_vectors.len() / 64) as f64) * 100.0;
            println!(
                "Video ID: {}, UUID: {}, Match Percentage: {:.2}%, Matching Scenes: {}",
                video_id, video.id, match_percentage, matches
            );
        }
    }
}

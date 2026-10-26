use ndarray::Array1;
mod idx;
mod network;
// turn bytes [0, 255] to [0.0, 1.0]
fn normalize_image(image: &[u8]) -> Array1<f32> {
    image.iter().map(|&val| val as f32 / 255.0).collect()
}

fn print_image(image: &[f32]) {
    const SHADES: &[u8] = b" .:-=+*#%@";
    for row in image.chunks(28) {
        let line: String = row
            .iter()
            .map(|&p| SHADES[(p * (SHADES.len() - 1) as f32) as usize] as char)
            .flat_map(|c| [c, c])
            .collect();
        println!("{line}");
    }
}

fn main() {
    let images = idx::load_images("data/train-images-idx3-ubyte/train-images.idx3-ubyte")
        .expect("failed to read training images");
    let labels = idx::load_labels("data/train-labels-idx1-ubyte/train-labels.idx1-ubyte")
        .expect("failed to read training labels");
    assert_eq!(images.count, labels.len());

    println!(
        "loaded {} images of {}x{}",
        images.count, images.rows, images.cols
    );
    for i in 0..100 {
        println!("\nlabel: {}", labels[i]);
        let data = images.get(i);
        print_image(&normalize_image(data).as_slice().unwrap());
    }
}

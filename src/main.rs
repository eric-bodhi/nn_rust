mod error;
mod gpu;
mod idx;
mod mat_autograd;
mod matrix;
mod train;

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

    let test_images = idx::load_images("data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte")
        .expect("failed to read test images");
    let test_labels = idx::load_labels("data/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte")
        .expect("failed to read test labels");

    train::train(&images, &labels, &test_images, &test_labels).expect("gpu training failed");
}

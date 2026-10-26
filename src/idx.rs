use std::fs;
use std::io;
use std::path::Path;

const IMAGE_MAGIC: u32 = 0x0000_0803; // u8 data, 3 dimensions
const LABEL_MAGIC: u32 = 0x0000_0801; // u8 data, 1 dimension

pub struct Images {
    pub count: usize,
    pub rows: usize,
    pub cols: usize,
    pub pixels: Vec<u8>,
}

impl Images {
    // Returns the pixels of image `i` as one flat slice of rows * cols bytes.
    pub fn get(&self, i: usize) -> &[u8] {
        let size = self.rows * self.cols;
        &self.pixels[i * size..(i + 1) * size]
    }
}

// IDX header numbers are 4-byte big-endian unsigned integers.
fn read_u32_be(bytes: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

pub fn load_images(path: impl AsRef<Path>) -> io::Result<Images> {
    let bytes = fs::read(path)?;
    assert!(bytes.len() >= 16, "file too short to be an IDX image file");
    assert_eq!(read_u32_be(&bytes, 0), IMAGE_MAGIC, "not an IDX image file");

    let count = read_u32_be(&bytes, 4) as usize;
    let rows = read_u32_be(&bytes, 8) as usize;
    let cols = read_u32_be(&bytes, 12) as usize;
    assert_eq!(
        bytes.len(),
        16 + count * rows * cols,
        "file size doesn't match header"
    );

    Ok(Images {
        count,
        rows,
        cols,
        pixels: bytes[16..].to_vec(),
    })
}

pub fn load_labels(path: impl AsRef<Path>) -> io::Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    assert!(bytes.len() >= 8, "file too short to be an IDX label file");
    assert_eq!(read_u32_be(&bytes, 0), LABEL_MAGIC, "not an IDX label file");

    let count = read_u32_be(&bytes, 4) as usize;
    assert_eq!(bytes.len(), 8 + count, "file size doesn't match header");

    Ok(bytes[8..].to_vec())
}

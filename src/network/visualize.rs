use super::{Activations, loss};

const SHADES: &[u8] = b" .:-=+*#%@";
const BAR_WIDTH: usize = 40;

impl Activations {
    // prints the forward pass top to bottom:
    // input image -> hidden layer -> output probabilities -> loss
    pub fn print_diagram(&self, label: u8) {
        let label = label as usize;

        println!("\n================ forward pass (label: {label}) ================");

        println!("\nINPUT  image: 784 pixels (28x28)");
        crate::print_image(self.x.as_slice().unwrap());

        println!("        |  weights1 (128x784) . image + bias1");
        println!("        v");

        let lowest = self.z1.fold(f32::INFINITY, |acc, &v| acc.min(v));
        let highest = self.z1.fold(f32::NEG_INFINITY, |acc, &v| acc.max(v));
        println!("HIDDEN pre-activation: 128 values, min {lowest:.2}, max {highest:.2}");

        println!("        |  relu (negatives become 0)");
        println!("        v");

        // 128 neurons drawn as 8 rows of 16, blank = off, brighter = bigger output
        let active = self.a1.iter().filter(|&&v| v > 0.0).count();
        let brightest = self.a1.fold(0_f32, |acc, &v| acc.max(v));
        println!("HIDDEN output: {active} of 128 neurons on (blank = off, brighter = larger)");
        println!("  +{}+", "-".repeat(32));
        for row in self.a1.as_slice().unwrap().chunks(16) {
            let line: String = row
                .iter()
                .map(|&v| {
                    if v > 0.0 && brightest > 0.0 {
                        // ceil so even a tiny positive output gets a visible mark
                        let index = ((v / brightest) * (SHADES.len() - 1) as f32).ceil() as usize;
                        SHADES[index] as char
                    } else {
                        ' '
                    }
                })
                .flat_map(|c| [c, c])
                .collect();
            println!("  |{line}|");
        }
        println!("  +{}+", "-".repeat(32));

        println!("        |  weights2 (10x128) . hidden output + bias2, then softmax");
        println!("        v");

        let prediction = self
            .p
            .iter()
            .enumerate()
            .fold(0, |best, (digit, &prob)| if prob > self.p[best] { digit } else { best });

        println!("OUTPUT probabilities (bar spans 0.0 to 1.0)");
        for (digit, &prob) in self.p.iter().enumerate() {
            let bar = "#".repeat((prob * BAR_WIDTH as f32).round() as usize);
            let marker = match (digit == label, digit == prediction) {
                (true, true) => "<- label, prediction",
                (true, false) => "<- label",
                (false, true) => "<- prediction",
                (false, false) => "",
            };
            println!("  {digit} |{bar:<BAR_WIDTH$}| {prob:.3} {marker}");
        }

        let verdict = if prediction == label { "correct" } else { "wrong" };
        println!("\nprediction: {prediction} ({verdict})");
        println!("LOSS  -ln(probabilities[{label}]) = {:.3}", loss(self.p[label]));
    }
}

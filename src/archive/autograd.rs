use super::network::Activations;

// automatic back propagation
#[derive(Debug)]
enum Op {
    Leaf,
    Add(usize, usize),
    Mul(usize, usize),
    Relu(usize),
}

#[derive(Debug)]
struct Node {
    value: f32,
    gradient: f64,
    operation: Op,
}

pub struct Tape {
    nodes: Vec<Node>,
}

pub struct ScalarNetwork {
    tape: Tape,
}

impl Tape {
    fn leaf(&mut self, value: f32) -> usize {
        self.nodes.push(Node {
            value: value,
            gradient: 0.0,
            operation: Op::Leaf,
        });
        self.nodes.len() - 1
    }

    fn mul(&mut self, a: usize, b: usize) -> usize {
        self.nodes.push(Node {
            value: self.nodes[a].value * self.nodes[b].value,
            gradient: 0.0,
            operation: Op::Mul(a, b),
        });

        self.nodes.len() - 1
    }

    fn add(&mut self, a: usize, b: usize) -> usize {
        self.nodes.push(Node {
            value: self.nodes[a].value + self.nodes[b].value,
            gradient: 0.0,
            operation: Op::Add(a, b),
        });
        self.nodes.len() - 1
    }

    fn relu(&mut self, z: usize) -> usize {
        self.nodes.push(Node {
            value: if self.nodes[z].value > 0.0 {
                self.nodes[z].value
            } else {
                0.0
            },
            gradient: 0.0,
            operation: Op::Relu(z),
        });
        self.nodes.len() - 1
    }

    fn back_prop(&mut self) {
        let length = self.nodes.len();
        self.nodes[length - 1].gradient = 1.0;
        for i in (0..self.nodes.len()).rev() {
            match self.nodes[i].operation {
                Op::Leaf => {}
                Op::Relu(z) => {
                    let relu_derivative = if self.nodes[z].value > 0.0 { 1.0 } else { 0.0 };
                    self.nodes[z].gradient += self.nodes[i].gradient * relu_derivative;
                }
                Op::Add(a, b) => {
                    self.nodes[a].gradient += self.nodes[i].gradient * 1.0;
                    self.nodes[b].gradient += self.nodes[i].gradient * 1.0;
                }
                Op::Mul(a, b) => {
                    self.nodes[a].gradient += self.nodes[i].gradient * self.nodes[b].value as f64;
                    self.nodes[b].gradient += self.nodes[i].gradient * self.nodes[a].value as f64;
                }
            };
        }
    }
}
// sample network: a single neuron, the same shape as one hidden unit in Network
//
//        x = 2.0 ──┐
//                  (*)── z1 = -6.0 ──┐
//       w = -3.0 ──┘                 (+)── z = 4.0 ──[relu]── a = 4.0   (a is the "loss" for now)
//                                    │
//       b = 10.0 ────────────────────┘
//
// forward:  z1 = x * w
//           z  = z1 + b
//           a  = relu(z)
//
// tape (pushed in forward order, every node comes after its inputs):
//
//   idx | name | op          | value | gradient (∂a/∂node)
//   ----+------+-------------+-------+--------------------
//    0  | x    | leaf        |   2.0 |   ?
//    1  | w    | leaf        |  -3.0 |   ?
//    2  | b    | leaf        |  10.0 |   ?
//    3  | z1   | x * w       |  -6.0 |   ?
//    4  | z    | z1 + b      |   4.0 |   ?
//    5  | a    | relu(z)     |   4.0 |   ?
//
// fill in the gradient column by hand before writing backward,
// then make backward produce the same numbers
pub fn autograd() {
    let mut sn = ScalarNetwork {
        tape: Tape { nodes: vec![] },
    };

    let x = sn.tape.leaf(2.0);
    let w = sn.tape.leaf(-3.0);
    let b = sn.tape.leaf(10.0);
    let z1 = sn.tape.mul(x, w);
    let z = sn.tape.add(z1, b);
    let a = sn.tape.relu(z);

    sn.tape.back_prop();
    println!("{:#?}", sn.tape.nodes);
    sn.tape.back_prop();
    println!("{:#?}", sn.tape.nodes);
}

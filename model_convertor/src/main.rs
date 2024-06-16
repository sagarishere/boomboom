use tensorflow::SavedModelBundle;

use tensorflow::Graph;
use tensorflow::Session;
use tensorflow::Tensor;

use pickle::{Pickler, Unpickler};
use std::fs::File;
use std::io::BufWriter;

use std::fs::File;
use std::io::BufReader;

fn main() {
    let mut graph = Graph::new();
    let config = SessionOptions::new();
    config.graph = Some(graph.clone());
    let mut session = Session::new(config, None).unwrap();
    // let mut session = Session::new(&graph).unwrap();
    let model = SavedModelBundle::load("../../results/models/my_own_model.h5", &["serve"]).unwrap();

    let mut file = BufWriter::new(File::create("../../results/models/my_own_model.pkl").unwrap());
    let mut pickler = Pickler::new(&mut file);
    pickler.pickle(&model).unwrap();

    let file = BufReader::new(File::open("../../results/models/my_own_model.pkl").unwrap());
    let mut unpickler = Unpickler::new(file);
    let loaded_model: SavedModelBundle = unpickler.load().unwrap();
}

use na::DMatrix;
use ply_rs::{parser, ply, ply::Property};
use std::io::BufReader;

fn extract_value(property: &Property) -> f64 {
    match *property {
        Property::Char(value) => value as f64,
        Property::UChar(value) => value as f64,
        Property::Short(value) => value as f64,
        Property::UShort(value) => value as f64,
        Property::Int(value) => value as f64,
        Property::UInt(value) => value as f64,
        Property::Float(value) => value as f64,
        Property::Double(value) => value as f64,
        _ => panic!("extract_value: Unexpected value found in property."),
    }
}

struct Vertex {
    xyz: [f64; 3],
    rgb: [u8; 3],
}

impl ply::PropertyAccess for Vertex {
    fn new() -> Self {
        Vertex {
            xyz: [0., 0., 0.],
            rgb: [0, 0, 0],
        }
    }

    fn set_property(&mut self, key: String, property: ply::Property) {
        match key.as_ref() {
            "x" => self.xyz[0] = extract_value(&property),
            "y" => self.xyz[1] = extract_value(&property),
            "z" => self.xyz[2] = extract_value(&property),
            "red" => self.rgb[0] = extract_value(&property) as u8,
            "green" => self.rgb[1] = extract_value(&property) as u8,
            "blue" => self.rgb[2] = extract_value(&property) as u8,
            _ => {}
        }
    }
}

pub fn read_point_cloud(path: &str) -> (DMatrix<f64>, DMatrix<u8>) {
    let file = std::fs::File::open(path).unwrap();
    let mut reader = BufReader::new(file);
    let parser = parser::Parser::<Vertex>::new();
    let header = parser.read_header(&mut reader).unwrap();
    let element = header
        .elements
        .get("vertex")
        .expect("Vertex element not found.");
    let vertex_vector = parser
        .read_payload_for_element(&mut reader, &element, &header)
        .expect("Failure when reading ply payload.");
    let mut points = DMatrix::zeros(vertex_vector.len(), 3);
    let mut colors = DMatrix::zeros(vertex_vector.len(), 3);
    vertex_vector.iter().enumerate().for_each(|(i, vertex)| {
        points.row_mut(i).copy_from_slice(&vertex.xyz);
        colors.row_mut(i).copy_from_slice(&vertex.rgb);
    });
    (points, colors)
}

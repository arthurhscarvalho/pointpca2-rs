use ply_rs::{parser, ply, ply::Property};
use std::io::BufReader;

fn extract_value(property: &Property) -> f32 {
    match *property {
        Property::Char(value) => value as f32,
        Property::UChar(value) => value as f32,
        Property::Short(value) => value as f32,
        Property::UShort(value) => value as f32,
        Property::Int(value) => value as f32,
        Property::UInt(value) => value as f32,
        Property::Float(value) => value as f32,
        Property::Double(value) => value as f32,
        _ => panic!("extract_value: Unexpected value found in property."),
    }
}

struct Vertex {
    xyz: [f32; 3],
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
            "red" => {
                self.rgb[0] = extract_value(&property) as u8;
            }
            "green" => {
                self.rgb[1] = extract_value(&property) as u8;
            }
            "blue" => {
                self.rgb[2] = extract_value(&property) as u8;
            }
            _ => {}
        }
    }
}

pub fn read_point_cloud(path: &str) -> (Vec<[f32; 3]>, Vec<[u8; 3]>) {
    let file = std::fs::File::open(path).unwrap();
    let mut reader = BufReader::new(file);
    let parser = parser::Parser::<Vertex>::new();
    let header = parser.read_header(&mut reader).unwrap();
    let element = header
        .elements
        .get("vertex")
        .expect("Vertex element not found.");
    let (points, colors) = parser
        .read_payload_for_element(&mut reader, &element, &header)
        .expect("Failure when reading ply payload.")
        .into_iter()
        .map(|Vertex { xyz, rgb }| (xyz, rgb))
        .unzip();
    (points, colors)
}

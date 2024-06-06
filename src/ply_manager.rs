use nalgebra::DMatrix;
use ply_rs::{parser, ply, ply::DefaultElement, ply::Property};

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

pub fn read_ply(path: &str) -> ply::Ply<DefaultElement> {
    let mut f = std::fs::File::open(path).unwrap();
    let p = parser::Parser::<ply::DefaultElement>::new();
    let ply = p.read_ply(&mut f);
    assert!(ply.is_ok());
    let ply = ply.unwrap();
    ply
}

pub fn read_ply_as_matrix(path: &str) -> (DMatrix<f64>, DMatrix<u8>) {
    let point_cloud = read_ply(path);
    let mut element_count = 0;
    for (_, element) in point_cloud.header.elements {
        if element_count > 0 {
            panic!("Multiple elements found in point_cloud header.");
        }
        element_count = element.count;
    }
    let mut points: DMatrix<f64> = DMatrix::zeros(element_count, 3);
    let mut colors: DMatrix<u8> = DMatrix::zeros(element_count, 3);
    for i in 0..element_count {
        let mut j = 0;
        let vertex = &point_cloud.payload["vertex"][i];
        for (_, property) in vertex {
            let value = extract_value(property);
            if j < 3 {
                points[(i, j)] = value;
            } else {
                colors[(i, j - 3)] = value as u8;
            }
            j += 1;
        }
    }
    (points, colors)
}

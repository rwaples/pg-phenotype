//! pg-phenotype composes pedigree-graph-core's public items from outside the
//! crate: validation, relationship pairs, and the pairwise kinship walk.

use pedigree_graph_core::graph::{self, Columns, Limits, SexEncoding};
use pedigree_graph_core::kinship::ancestry::AncestorSignatures;
use pedigree_graph_core::kinship::pairwise::Walker;
use pedigree_graph_core::kinship::KinshipPedigree;
use pedigree_graph_core::relationships::{
    pair_blocks_compact, Category, CategorySet, Execution, MaxDegree, Pedigree, Progress,
};
use pedigree_graph_core::topology;

#[test]
fn build_pairs_and_walk_compose_from_outside_the_crate() {
    // Founders 10 and 11; full sibs 12 and 13; 14 is 12's child by an
    // external father 99.
    let ids = [10i64, 11, 12, 13, 14];
    let mother = [-1i64, -1, 10, 10, 12];
    let father = [-1i64, -1, 11, 11, 99];
    let built = graph::build(
        Columns {
            ids: &ids,
            mother: &mother,
            father: &father,
            twin: None,
            sex: None,
            generation: None,
            birth_year: None,
        },
        SexEncoding::Simace,
        Limits::default(),
    )
    .expect("valid pedigree");

    let ped = Pedigree::try_new(
        &built.mother_rows,
        &built.father_rows,
        &built.twin_rows,
        &built.mother_ids,
        &built.father_ids,
    )
    .expect("engine input");
    let view: Vec<i32> = (0..ids.len() as i32).collect();
    let blocks = pair_blocks_compact(
        &ped,
        MaxDegree::try_new(2).expect("degree"),
        CategorySet::up_to_degree(2),
        &view,
        Execution::Speed,
        &Progress::default(),
    )
    .expect("pairs");
    assert_eq!(blocks.get(Category::FS).len(), 1);
    assert_eq!(blocks.get(Category::Av).len(), 1);

    let depth = topology::structural_depth(&built.mother_rows, &built.father_rows);
    let kped = KinshipPedigree::try_new(
        &built.mother_rows,
        &built.father_rows,
        &built.twin_rows,
        &depth,
    )
    .expect("kinship input");
    let signatures = AncestorSignatures::build(&kped).expect("signatures");
    let mut walker = Walker::new(kped, &signatures).expect("walker");
    assert_eq!(walker.resolve(2, 3).expect("sibs"), 0.25);
    assert_eq!(walker.resolve(3, 4).expect("aunt"), 0.125);
    assert_eq!(walker.resolve(0, 1).expect("founders"), 0.0);
}

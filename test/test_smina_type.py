from data.pdbbind_dataset import GridPDB, AtomData

def test_smina_type():
    ligfile = 'test/fixtures/4rdn_ligand.pdbqt'
    lig = GridPDB(ligfile)
    atom_data = AtomData()
    ref_atom_types = [4, 4, 4, 4, 14, 7, 6, 10, 6, 10, 6, 6, 10, 6, 8, 2, 4, 13, 2, 13, 2, 4, 13, 2]
    test_atom_types = []
    for i in range(lig.natom):
        data_i = atom_data[lig.atomdata[i]]
        test_atom_types.append(data_i['smina_type'])
    assert ref_atom_types == test_atom_types

def test_smina_type_adp():
    ligfile = 'test/fixtures/adp.pdbqt'
    lig = GridPDB(ligfile)
    atom_data = AtomData()
    ref_atom_types = [4, 4, 4, 4, 14, 7, 6, 10, 6, 10, 6, 8, 2, 2, 6, 10, 6, 13, 2, 13, 2, 4, 14, 17, 14, 13, 2, 14, 17, 14, 13, 2, 13, 2]
    test_atom_types = []
    for i in range(lig.natom):
        data_i = atom_data[lig.atomdata[i]]
        test_atom_types.append(data_i['smina_type'])
    assert ref_atom_types == test_atom_types
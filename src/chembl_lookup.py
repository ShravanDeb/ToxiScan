# src/chembl_lookup.py
# Call this anytime you need ChEMBL info for a specific compound

from chembl_webresource_client.new_client import new_client

def lookup_compound(smiles):
    """
    Give it a SMILES string → get back ChEMBL info for that compound.
    Returns a dict with name, max potency, and known assays.
    Returns None if compound not found in ChEMBL.
    """
    molecule = new_client.molecule
    activity = new_client.activity

    # Step 1 — find the compound by SMILES
    results = molecule.filter(molecule_structures__canonical_smiles=smiles)
    if not results:
        return None

    chembl_id = results[0]["molecule_chembl_id"]
    name      = results[0].get("pref_name", "Unknown")

    # Step 2 — get its bioactivity records
    acts = activity.filter(
        molecule_chembl_id=chembl_id,
        pchembl_value__isnull=False
    ).only(["pchembl_value", "standard_type", "assay_description"])[:10]

    acts_list = list(acts)

    return {
        "chembl_id"  : chembl_id,
        "name"       : name,
        "num_assays" : len(acts_list),
        "max_potency": max([float(a["pchembl_value"]) for a in acts_list], default=0),
        "assays"     : acts_list
    }


def lookup_by_name(drug_name):
    """
    Give it a drug name like 'Aspirin' → get back ChEMBL info.
    Useful for the batch screener.
    """
    molecule = new_client.molecule
    results  = molecule.filter(pref_name__iexact=drug_name)
    if not results:
        return None

    chembl_id = results[0]["molecule_chembl_id"]
    name      = results[0].get("pref_name", drug_name)
    smiles    = results[0].get("molecule_structures", {})
    smiles    = smiles.get("canonical_smiles", "") if smiles else ""

    return {
        "chembl_id": chembl_id,
        "name"     : name,
        "smiles"   : smiles,
    }
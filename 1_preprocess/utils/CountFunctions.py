def count_cbc_cases(data):
    comp_data = data.query("~(f__WBC.isnull() & f__HGB.isnull() & f__MCV.isnull() & f__PLT.isnull() & f__RBC.isnull())",
                           engine='python')
    unique_data = comp_data.drop_duplicates(subset=["Id", "Center"])
    return len(unique_data)


def count_cbc(data):
    comp_data = data.query("~(f__WBC.isnull() & f__HGB.isnull() & f__MCV.isnull() & f__PLT.isnull() & f__RBC.isnull())",
                           engine='python')
    return len(comp_data)
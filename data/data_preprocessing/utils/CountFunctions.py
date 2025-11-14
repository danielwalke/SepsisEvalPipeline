def count_cbc_cases(data):
    comp_data = data.query("~(WBC.isnull() & HGB.isnull() & MCV.isnull() & PLT.isnull() & RBC.isnull())",
                           engine='python')
    unique_data = comp_data.drop_duplicates(subset=["Id", "Center"])
    return len(unique_data)


def count_cbc(data):
    comp_data = data.query("~(WBC.isnull() & HGB.isnull() & MCV.isnull() & PLT.isnull() & RBC.isnull())",
                           engine='python')
    return len(comp_data)
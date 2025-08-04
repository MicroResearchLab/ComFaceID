import pandas as pd

def search_one_by_one(csv,Id_name,mid):
    df = pd.read_csv(csv)
    result_df = df[df[Id_name]==mid]
    dicts = result_df.to_dict('records')[0]
    return dicts

class SelectAllData:
    
    def __init__(self,csv_file_path):
        self.all_data_df = pd.read_csv(csv_file_path)

    def search_one(self,Id_name,mid):
        result_df = self.all_data_df[self.all_data_df[Id_name]==mid]
        if len(result_df.to_dict('records')) >0:
            dicts = result_df.to_dict('records')[0]
            return dicts
        else:
            raise(mid," can not found")

    def search_from_list(self,Id_name,mids):
        result_df = self.all_data_df[self.all_data_df[Id_name].isin(mids)]
        dicts = result_df.to_dict('records')
        return dicts
    

class loadDatabase(SelectAllData):
    def __init__(self,path):
        super().__init__(path)
    def search_one(self,mid):
        return super().search_one('Spectrum_ID',mid)
    def search_from_list(self,mids):
        return super().search_from_list('Spectrum_ID',mids)
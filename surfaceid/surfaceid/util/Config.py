import yaml

class Config:
    def __init__(self,path):
        self.config = yaml.safe_load(open(path, "r"))
        self.CONTACT = self.config.get('CONTACT')
        self.DESC = self.config.get('DESC')
        self.SEARCH = self.config.get('SEARCH')
        self.ALIGN = self.config.get('ALIGN')
        self.SAVEPLY = self.config.get('SAVEPLY')
        self.HEATMAP = self.config.get('HEATMAP')
        self.REMOVE_OLD = self.ALIGN and self.SAVEPLY and self.SEARCH
        self.CATALOG_PAIRS = self.config.get('PATH').get('CATALOG_PAIRS')
        self.expand_radius = self.config.get('SPATIAL_PARAMETER').get('expand_radius')
        self.neighbor_dist = self.config.get('SPATIAL_PARAMETER').get('neighbor_dist')
        self.nmin_pts = self.config.get('SPATIAL_PARAMETER').get('nmin_pts')
        self.nmin_pts_library = self.config.get('SPATIAL_PARAMETER').get('nmin_pts_library')
        self.prefix = self.config.get('MODEL').get('NAME')
        self.case = self.config.get('PATH').get('CASE')
        self.OUTDIR_RESULTS = f"{self.case}_results"
        self.OUTDIR = self.config.get('PATH').get('OUTDIR')
        self.target = self.config.get('PATH').get('TARGET')
        self.contact_thres1 = self.config.get('SPATIAL_PARAMETER').get('contact_thres1')  # for identifying iface points
        self.contact_thres2 =  self.config.get('SPATIAL_PARAMETER').get('contact_thres2') # for distance based interacting points
        self.contact_mode = self.config.get('SPATIAL_PARAMETER').get('contact_mode')
        self.npatience = self.config.get('SPATIAL_PARAMETER').get('npatience')
        self.thres = self.config.get('SPATIAL_PARAMETER').get('thres')
        self.savedir = self.config.get('PATH').get('SAVEDIR')
        self.params = self.config.get('MODEL_PARAMETER')
        self.hitcolumns = self.config.get('HIT_COLUMNS')
import tqdm
import novelpy

#%% cooc

concept_cooc = novelpy.utils.cooc_utils.create_cooc(
                client_name="mongodb://localhost:27017",
                db_name = "UBI",    
                collection_name = "works_UBI_global",
                year_var="publication_year",
                var = "concepts",
                sub_var = "display_name",
                weighted_network = True, self_loop = True)

concept_cooc.main()

concept_cooc = novelpy.utils.cooc_utils.create_cooc(
                client_name="mongodb://localhost:27017",
                db_name = "UBI",    
                collection_name = "works_UBI_global",
                year_var="publication_year",
                var = "concepts",
                sub_var = "display_name",
                weighted_network = False, self_loop = False)

concept_cooc.main()


#%% Novelty indicators



# Foster et al.(2015) Meshterms_sample
for focal_year in tqdm.tqdm(range(1950,2021), desc = "Computing indicator for window of time"):
    Foster = novelpy.indicators.Foster2015( client_name="mongodb://localhost:27017",
                                       db_name = "UBI",    
                                       collection_name = "works_UBI_global",
                                       id_variable = 'id_cleaned',
                                       year_variable = 'publication_year',
                                       variable = "concepts",
                                       sub_variable = "display_name",
                                       focal_year = focal_year,
                                       community_algorithm = "Louvain",
                                       density = True)
    Foster.get_indicator()


# Lee et al.(2015) Meshterms_sample
for focal_year in tqdm.tqdm(range(1950,2021), desc = "Computing indicator for window of time"):
    Lee = novelpy.indicators.Lee2015(client_name="mongodb://localhost:27017",
                                       db_name = "UBI",    
                                       collection_name = "works_UBI_global",
                                       id_variable = 'id_cleaned',
                                       year_variable = 'publication_year',
                                       variable = "concepts",
                                       sub_variable = "display_name",
                                        focal_year = focal_year,
                                        density = True)
    Lee.get_indicator()

# Wang et al.(2017) Meshterms_sample
for focal_year in tqdm.tqdm(range(1950,2021)):
    Wang = novelpy.indicators.Wang2017( client_name="mongodb://localhost:27017",
                                        db_name = "UBI",    
                                        collection_name = "works_UBI_global",
                                        id_variable = 'id_cleaned',
                                        year_variable = 'publication_year',
                                        variable = "concepts",
                                        sub_variable = "display_name",
                                        focal_year = focal_year,
                                        time_window_cooc = 3,
                                        n_reutilisation = 1,
                                        starting_year = 1995,
                                        density = True)
    Wang.get_indicator()



#%% Disruptiveness Indicators

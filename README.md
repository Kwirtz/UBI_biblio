# Sixty years of Basic Income research (Work in Progress)

This repository host codes from the paper Sixty years of Basic Income research (Jacob and Wirtz, 2025).

## Aim of the paper

This article presents a quantitative history of basic income (BI) research within the Social Sciences from the 1960s to the present, utilizing bibliometric analysis on OpenAlex data. We identify five main research communities; Social Justice, Experiment, Tax and Labor Supply, Degrowth, and Others, and four major international collaboration clusters. Through this framework, we identify three major periods in BI research; an early experimental focus (1960–1980), a shift toward taxation, labor supply, and social justice (1980–2000), and a recent diversification into ecological concerns, thinking on social protection in South Africa and Germany, and care economics (2000–2020). A key insight from our study is the enduring influence of Negative Income Tax (NIT) and Minimum Income Guarantee (MIG) within BI research. Although the conceptual boundaries of BI have expanded to include broader social justice and ecological perspectives, the Experiment and Tax/Labor Supply communities continue to engage deeply with NIT and MIG. This persistence reflects long-standing research traditions, underscoring the distinct policy concerns shaping different strands of BI research. Ultimately, our study deepens our understanding of BI as an evolving research field, shaped by distinct intellectual traditions, regional specializations, and shifting policy priorities over time.

## Reproduction step

### Downloads

The first step was to retrieve the data. We took a snapshot of OpenAlex and processed it using the scripts in Downloads/0-4 to generate our sample dataset.

Since the dataset is relatively small, all data from these steps is already available in the Data/ directory.

If you want to skip the data processing steps, you can use the final dataset:

📂 Data/UBI.works_UBI_gobu_2.zip

Simply unzip the file and import it into MongoDB:

    Database name: UBI
    Collection name: works_UBI_gobu_2

Once imported, you can proceed with the scripts in Downloads/5-7. Alternatively, if you prefer, the output data from these steps is already available in the Data/ folder.



### Analysis

We then processed the dataset further using scripts 1-4 in the Analysis/ directory. These scripts generated the final dataset used in both the figures (Analysis/) and statistical analyses (Stats/).

However, if you prefer to skip this step, all required data is already stored in Data/, allowing you to directly:

    Generate figures
    Run statistical analyses (Stats/)
    Explore the final figures and tables in Results/

This ensures you have everything needed without reprocessing the data. 
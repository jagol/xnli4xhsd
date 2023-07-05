# export data_dir=$(cat paths.json | jq -r '.data_dir')
mkdir "$data_dir" "$data_dir/HateCheck" "$data_dir/BAS19_ES" "$data_dir/FOR19_PT" "$data_dir/FOU18_EN" "$data_dir/HAS19_HI" "$data_dir/HAS20_HI" "$data_dir/HAS21_HI" "$data_dir/KEN20_EN" "$data_dir/OUS19_AR" "$data_dir/OUS19_FR" "$data_dir/SAN20_IT" "$data_dir/MHC" 

wget -O "$data_dir/HateCheck/HateCheck_test.csv" https://raw.githubusercontent.com/paul-rottger/hatecheck-data/main/test_suite_cases.csv
wget -O "$data_dir/ETHOS_Binary/Ethos_Dataset_Binary.csv" https://raw.githubusercontent.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/master/ethos/ethos_data/Ethos_Dataset_Binary.csv

wget -O "$data_dir/BAS19_ES/train.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/bas19_es/train_4100.csv 
wget -O "$data_dir/BAS19_ES/test.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/bas19_es/test_2000.csv 
wget -O "$data_dir/BAS19_ES/dev.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/bas19_es/dev_500.csv 

wget -O "$data_dir/DYN21_EN/train.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/dyn21_en/train_38644.csv 
wget -O "$data_dir/DYN21_EN/test.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/dyn21_en/test_2000.csv 
wget -O "$data_dir/DYN21_EN/dev.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/dyn21_en/dev_500.csv 

wget -O "$data_dir/FOR19_PT/train.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/for19_pt/train_3170.csv 
wget -O "$data_dir/FOR19_PT/test.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/for19_pt/test_2000.csv 
wget -O "$data_dir/FOR19_PT/dev.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/for19_pt/dev_500.csv 

wget -O "$data_dir/FOU18_EN/train.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/fou18_en/train_20065.csv 
wget -O "$data_dir/FOU18_EN/test.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/fou18_en/test_2000.csv 
wget -O "$data_dir/FOU18_EN/dev.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/fou18_en/dev_500.csv 

wget -O "$data_dir/HAS19_HI/train.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/has19_hi/train_4165.csv 
wget -O "$data_dir/HAS19_HI/test.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/has19_hi/test_1318.csv 
wget -O "$data_dir/HAS19_HI/dev.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/has19_hi/dev_500.csv 

wget -O "$data_dir/HAS20_HI/train.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/has20_hi/train_2463.csv 
wget -O "$data_dir/HAS20_HI/test.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/has20_hi/test_1269.csv 
wget -O "$data_dir/HAS20_HI/dev.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/has20_hi/dev_500.csv 

wget -O "$data_dir/HAS21_HI/train.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/has21_hi/train_2094.csv 
wget -O "$data_dir/HAS21_HI/test.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/has21_hi/test_2000.csv 
wget -O "$data_dir/HAS21_HI/dev.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/has21_hi/dev_500.csv 

wget -O "$data_dir/KEN20_EN/train.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/ken20_en/train_20692.csv 
wget -O "$data_dir/KEN20_EN/test.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/ken20_en/test_2000.csv 
wget -O "$data_dir/KEN20_EN/dev.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/ken20_en/dev_500.csv 

wget -O "$data_dir/OUS19_AR/train.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/ous19_ar/train_2053.csv 
wget -O "$data_dir/OUS19_AR/test.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/ous19_ar/test_1000.csv 
wget -O "$data_dir/OUS19_AR/dev.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/ous19_ar/dev_300.csv 

wget -O "$data_dir/OUS19_FR/train.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/ous19_fr/train_2014.csv 
wget -O "$data_dir/OUS19_FR/test.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/ous19_fr/test_1500.csv 
wget -O "$data_dir/OUS19_FR/dev.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/ous19_fr/dev_500.csv 

wget -O "$data_dir/SAN20_IT/train.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/san20_it/train_5600.csv 
wget -O "$data_dir/SAN20_IT/test.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/san20_it/test_2000.csv 
wget -O "$data_dir/SAN20_IT/dev.csv" https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/master/0_data/main/1_clean/san20_it/dev_500.csv 

wget -O "$data_dir/MHC/hatecheck_cases_final_arabic.csv" https://raw.githubusercontent.com/rewire-online/multilingual-hatecheck/main/MHC%20Final%20Cases/hatecheck_cases_final_arabic.csv
wget -O "$data_dir/MHC/hatecheck_cases_final_dutch.csv" https://raw.githubusercontent.com/rewire-online/multilingual-hatecheck/main/MHC%20Final%20Cases/hatecheck_cases_final_dutch.csv
wget -O "$data_dir/MHC/hatecheck_cases_final_french.csv" https://raw.githubusercontent.com/rewire-online/multilingual-hatecheck/main/MHC%20Final%20Cases/hatecheck_cases_final_french.csv
wget -O "$data_dir/MHC/hatecheck_cases_final_german.csv" https://raw.githubusercontent.com/rewire-online/multilingual-hatecheck/main/MHC%20Final%20Cases/hatecheck_cases_final_german.csv
wget -O "$data_dir/MHC/hatecheck_cases_final_hindi.csv" https://raw.githubusercontent.com/rewire-online/multilingual-hatecheck/main/MHC%20Final%20Cases/hatecheck_cases_final_hindi.csv
wget -O "$data_dir/MHC/hatecheck_cases_final_italian.csv" https://raw.githubusercontent.com/rewire-online/multilingual-hatecheck/main/MHC%20Final%20Cases/hatecheck_cases_final_italian.csv
wget -O "$data_dir/MHC/hatecheck_cases_final_mandarin.csv" https://raw.githubusercontent.com/rewire-online/multilingual-hatecheck/main/MHC%20Final%20Cases/hatecheck_cases_final_mandarin.csv
wget -O "$data_dir/MHC/hatecheck_cases_final_polish.csv" https://raw.githubusercontent.com/rewire-online/multilingual-hatecheck/main/MHC%20Final%20Cases/hatecheck_cases_final_polish.csv
wget -O "$data_dir/MHC/hatecheck_cases_final_portuguese.csv" https://raw.githubusercontent.com/rewire-online/multilingual-hatecheck/main/MHC%20Final%20Cases/hatecheck_cases_final_portuguese.csv
wget -O "$data_dir/MHC/hatecheck_cases_final_spanish.csv" https://raw.githubusercontent.com/rewire-online/multilingual-hatecheck/main/MHC%20Final%20Cases/hatecheck_cases_final_spanish.csv

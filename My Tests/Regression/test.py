import os
import torch
os.environ["KERAS_BACKEND"] = "torch"
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam


df1 = pd.read_csv("/Users/talalkhan/Documents/Data Sets/Second Challange/train.csv")
df2 = pd.read_csv("/Users/talalkhan/Documents/Data Sets/Second Challange/test.csv")
row_ids = df2['row ID']
df1 = df1.drop(columns=['sub_area'])
df2 = df2.drop(columns=['sub_area','row ID'])


#categorical_columns = df1.select_dtypes(include=['object']).columns

# Label encode categorical columns
#label_encoder = LabelEncoder()
#for col in categorical_columns:
#   df1[col] = label_encoder.fit_transform(df1[col])
#   df2[col] = label_encoder.transform(df2[col])


#df1_encoded = pd.get_dummies(df1)
#df2_encoded = pd.get_dummies(df2)



categorical_columns = df1.select_dtypes(include=['object']).columns

# Label encode categorical columns
label_encoder = LabelEncoder()
for col in categorical_columns:
   df1[col] = label_encoder.fit_transform(df1[col])
   df2[col] = label_encoder.transform(df2[col])

#df1_encoded = df1_encoded.drop(columns=['area_m', 'raion_popul', 'green_zone_part', 'indust_part', 'children_preschool', 'preschool_education_centers_raion', 'children_school', 'school_education_centers_raion', 'healthcare_centers_raion', 'sport_objects_raion', 'additional_education_raion', 'shopping_centers_raion', 'full_all', 'male_f', 'female_f', 'young_all', 'young_male', 'young_female', 'work_all', 'work_male', 'work_female', 'ekder_all', 'ekder_male', 'ekder_female', '0_6_all', '0_6_male', '0_6_female', '7_14_all', '7_14_male', '7_14_female', '0_17_all', '0_17_male', '0_17_female', '16_29_all', '16_29_male', '16_29_female', '0_13_all', '0_13_male', '0_13_female', 'raion_build_count_with_material_info', 'build_count_block', 'build_count_wood', 'build_count_frame', 'build_count_brick', 'build_count_panel', 'build_count_mix', 'raion_build_count_with_builddate_info', 'build_count_1921-1945', 'build_count_1946-1970', 'build_count_1971-1995', 'ID_metro', 'metro_min_avto', 'metro_km_avto', 'metro_min_walk', 'metro_km_walk', 'park_km', 'green_zone_km', 'water_treatment_km', 'cemetery_km', 'incineration_km', 'railroad_station_walk_km', 'railroad_station_walk_min', 'ID_railroad_station_walk', 'railroad_station_avto_km', 'railroad_station_avto_min', 'ID_railroad_station_avto', 'water_km', 'mkad_km', 'ttk_km', 'sadovoe_km', 'bulvar_ring_km', 'kremlin_km', 'big_road1_km', 'ID_big_road1', 'big_road2_km', 'ID_big_road2', 'railroad_km', 'zd_vokzaly_avto_km', 'ID_railroad_terminal', 'bus_terminal_avto_km', 'ID_bus_terminal', 'oil_chemistry_km', 'nuclear_reactor_km', 'radiation_km', 'power_transmission_line_km', 'thermal_power_plant_km', 'ts_km', 'big_market_km', 'market_shop_km', 'swim_pool_km', 'ice_rink_km', 'stadium_km', 'basketball_km', 'hospice_morgue_km', 'detention_facility_km', 'university_km', 'workplaces_km', 'shopping_centers_km', 'office_km', 'mosque_km', 'theater_km', 'museum_km', 'exhibition_km', 'catering_km', 'green_part_500', 'green_part_1000', 'prom_part_1000', 'green_part_1500', 'prom_part_1500', 'sport_count_1500', 'market_count_1500', 'green_part_2000', 'prom_part_2000', 'trc_count_2000', 'cafe_sum_2000_min_price_avg', 'cafe_sum_2000_max_price_avg', 'cafe_avg_price_2000', 'mosque_count_2000', 'sport_count_2000', 'market_count_2000', 'green_part_3000', 'prom_part_3000', 'office_sqm_3000', 'trc_count_3000', 'trc_sqm_3000', 'cafe_sum_3000_min_price_avg', 'cafe_sum_3000_max_price_avg', 'cafe_avg_price_3000', 'mosque_count_3000', 'sport_count_3000', 'market_count_3000', 'green_part_5000', 'prom_part_5000', 'office_count_5000', 'office_sqm_5000', 'trc_count_5000', 'trc_sqm_5000', 'cafe_count_5000', 'cafe_sum_5000_min_price_avg', 'cafe_sum_5000_max_price_avg', 'cafe_avg_price_5000', 'cafe_count_5000_na_price', 'cafe_count_5000_price_500', 'cafe_count_5000_price_1000', 'cafe_count_5000_price_1500', 'cafe_count_5000_price_2500', 'cafe_count_5000_price_4000', 'cafe_count_5000_price_high', 'big_church_count_5000', 'church_count_5000', 'mosque_count_5000', 'leisure_count_5000', 'sport_count_5000', 'market_count_5000', 'product_type_OwnerOccupier', 'culture_objects_top_25_yes', 'thermal_power_plant_raion_yes', 'incineration_raion_yes', 'oil_chemistry_raion_yes', 'radiation_raion_yes', 'railroad_terminal_raion_yes', 'big_market_raion_yes', 'nuclear_reactor_raion_yes', 'detention_facility_raion_yes', 'water_1line_yes', 'big_road1_1line_yes', 'railroad_1line_yes', 'ecology_good', 'ecology_no data', 'ecology_poor', 'ecology_satisfactory'],axis=1)
#df2_encoded = df2_encoded.drop(columns=['area_m', 'raion_popul', 'green_zone_part', 'indust_part', 'children_preschool', 'preschool_education_centers_raion', 'children_school', 'school_education_centers_raion', 'healthcare_centers_raion', 'sport_objects_raion', 'additional_education_raion', 'shopping_centers_raion', 'full_all', 'male_f', 'female_f', 'young_all', 'young_male', 'young_female', 'work_all', 'work_male', 'work_female', 'ekder_all', 'ekder_male', 'ekder_female', '0_6_all', '0_6_male', '0_6_female', '7_14_all', '7_14_male', '7_14_female', '0_17_all', '0_17_male', '0_17_female', '16_29_all', '16_29_male', '16_29_female', '0_13_all', '0_13_male', '0_13_female', 'raion_build_count_with_material_info', 'build_count_block', 'build_count_wood', 'build_count_frame', 'build_count_brick', 'build_count_panel', 'build_count_mix', 'raion_build_count_with_builddate_info', 'build_count_1921-1945', 'build_count_1946-1970', 'build_count_1971-1995', 'ID_metro', 'metro_min_avto', 'metro_km_avto', 'metro_min_walk', 'metro_km_walk', 'park_km', 'green_zone_km', 'water_treatment_km', 'cemetery_km', 'incineration_km', 'railroad_station_walk_km', 'railroad_station_walk_min', 'ID_railroad_station_walk', 'railroad_station_avto_km', 'railroad_station_avto_min', 'ID_railroad_station_avto', 'water_km', 'mkad_km', 'ttk_km', 'sadovoe_km', 'bulvar_ring_km', 'kremlin_km', 'big_road1_km', 'ID_big_road1', 'big_road2_km', 'ID_big_road2', 'railroad_km', 'zd_vokzaly_avto_km', 'ID_railroad_terminal', 'bus_terminal_avto_km', 'ID_bus_terminal', 'oil_chemistry_km', 'nuclear_reactor_km', 'radiation_km', 'power_transmission_line_km', 'thermal_power_plant_km', 'ts_km', 'big_market_km', 'market_shop_km', 'swim_pool_km', 'ice_rink_km', 'stadium_km', 'basketball_km', 'hospice_morgue_km', 'detention_facility_km', 'university_km', 'workplaces_km', 'shopping_centers_km', 'office_km', 'mosque_km', 'theater_km', 'museum_km', 'exhibition_km', 'catering_km', 'green_part_500', 'green_part_1000', 'prom_part_1000', 'green_part_1500', 'prom_part_1500', 'sport_count_1500', 'market_count_1500', 'green_part_2000', 'prom_part_2000', 'trc_count_2000', 'cafe_sum_2000_min_price_avg', 'cafe_sum_2000_max_price_avg', 'cafe_avg_price_2000', 'mosque_count_2000', 'sport_count_2000', 'market_count_2000', 'green_part_3000', 'prom_part_3000', 'office_sqm_3000', 'trc_count_3000', 'trc_sqm_3000', 'cafe_sum_3000_min_price_avg', 'cafe_sum_3000_max_price_avg', 'cafe_avg_price_3000', 'mosque_count_3000', 'sport_count_3000', 'market_count_3000', 'green_part_5000', 'prom_part_5000', 'office_count_5000', 'office_sqm_5000', 'trc_count_5000', 'trc_sqm_5000', 'cafe_count_5000', 'cafe_sum_5000_min_price_avg', 'cafe_sum_5000_max_price_avg', 'cafe_avg_price_5000', 'cafe_count_5000_na_price', 'cafe_count_5000_price_500', 'cafe_count_5000_price_1000', 'cafe_count_5000_price_1500', 'cafe_count_5000_price_2500', 'cafe_count_5000_price_4000', 'cafe_count_5000_price_high', 'big_church_count_5000', 'church_count_5000', 'mosque_count_5000', 'leisure_count_5000', 'sport_count_5000', 'market_count_5000', 'product_type_OwnerOccupier', 'culture_objects_top_25_yes', 'thermal_power_plant_raion_yes', 'incineration_raion_yes', 'oil_chemistry_raion_yes', 'radiation_raion_yes', 'railroad_terminal_raion_yes', 'big_market_raion_yes', 'nuclear_reactor_raion_yes', 'detention_facility_raion_yes', 'water_1line_yes', 'big_road1_1line_yes', 'railroad_1line_yes', 'ecology_good', 'ecology_no data', 'ecology_poor', 'ecology_satisfactory'],axis=1)
print(df2.shape)


X = df1.drop(columns=['price_doc',],axis=1)
y = df1['price_doc']



imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
df2_encoded = imputer.fit_transform(df2)


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
df2_encoded_scaled = scaler.fit_transform(df2_encoded)


#X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
#y_train = y_train.astype('float32')
#y_test = y_test.astype('float32')

X_scaled = X_scaled.astype('float32')
df2_encoded_scaled = df2_encoded_scaled.astype('float32')



n_features = X_scaled.shape[1]
# Create a Sequential model
model = Sequential()
model.add(Dense(100, input_dim=n_features, activation='relu'))

model.add(Dense(80, activation='sigmoid'))

model.add(Dense(60, activation='relu'))

model.add(Dense(40, activation='sigmoid'))

model.add(Dense(20, activation='relu'))

model.add(Dense(10, activation='relu'))

model.add(Dense(1))

from keras.optimizers import Adam
optimizer = Adam(learning_rate=0.005)

# Compile the model
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

#early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)



#print(model.summary())

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(X_scaled, y, epochs=200, batch_size=256 ,callbacks=[early_stopping], validation_split=0.2)

pred = model.predict(df2_encoded_scaled)


print(pred)
#print(mean_squared_error(y_test,pred))


# Combine test row IDs with their corresponding predictions into a DataFrame
output = pd.DataFrame({'row ID': row_ids, 'price_doc': pred.flatten()})

# Output the DataFrame to a CSV file
output.to_csv('submission72_25253.csv', index=False)
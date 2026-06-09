import numpy as np
import pandas as pd

HVAC_MAP = {
    'cooling_active': 'cooling_active', 'COOLING_ACTIVE': 'cooling_active',
    'Cooling_Active': 'cooling_active', 'Cooling_active': 'cooling_active',
    'heating_active': 'heating_active', 'HEATING_ACTIVE': 'heating_active',
    'Heating_Active': 'heating_active', 'Heating_active': 'heating_active',
    'maintenance_mode': 'maintenance_mode', 'MAINTENANCE_MODE': 'maintenance_mode',
    'Maintenance_Mode': 'maintenance_mode', 'Maintenance_mode': 'maintenance_mode',
    'eco_mode': 'eco_mode', 'ECO_MODE': 'eco_mode', 'Eco_Mode': 'eco_mode',
    'Eco_mode': 'eco_mode', 'Eco': 'eco_mode',
    'off': 'off', 'OFF': 'off', 'Off': 'off',
    'ventilation_only': 'ventilation_only', 'VENTILATION_ONLY': 'ventilation_only',
    'Ventilation_Only': 'ventilation_only', 'Ventilation_only': 'ventilation_only',
}

ACTIVITY_MAP = {
    'Low Activity': 'Low Activity', 'Low_Activity': 'Low Activity',
    'LowActivity': 'Low Activity',
    'Moderate Activity': 'Moderate Activity', 'ModerateActivity': 'Moderate Activity',
    'High Activity': 'High Activity',
}


def clean_gas_monitoring(df: pd.DataFrame) -> pd.DataFrame:
    # Applies cleaning rules from eda.ipynb: caps outliers, imputes missing values, standardises labels
    df.loc[df['Temperature'] > 60, 'Temperature'] = np.nan
    df['Temperature'] = df['Temperature'].fillna(df['Temperature'].median())

    df.loc[df['Humidity'] < 0, 'Humidity'] = np.nan
    df.loc[df['Humidity'] > 100, 'Humidity'] = np.nan
    df['Humidity'] = df['Humidity'].fillna(df['Humidity'].median())

    df.loc[df['CO2_InfraredSensor'] < 0, 'CO2_InfraredSensor'] = np.nan
    df['CO2_InfraredSensor'] = df['CO2_InfraredSensor'].fillna(df['CO2_InfraredSensor'].median())

    df['CO_GasSensor'] = df['CO_GasSensor'].fillna(df['CO_GasSensor'].mode()[0]).astype(int)

    df['MetalOxideSensor_Unit2'] = df['MetalOxideSensor_Unit2'].fillna(df['MetalOxideSensor_Unit2'].median())

    df['HVAC Operation Mode'] = df['HVAC Operation Mode'].map(HVAC_MAP).fillna('other')

    df['Ambient Light Level'] = df['Ambient Light Level'].fillna(df['Ambient Light Level'].mode()[0])

    df['Activity Level'] = df['Activity Level'].map(ACTIVITY_MAP)

    return df

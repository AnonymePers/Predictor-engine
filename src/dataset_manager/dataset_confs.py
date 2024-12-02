import os

case_id_col = {}
activity_col = {}
resource_col = {}
timestamp_col = {}
label_col = {}
pos_label = {}
neg_label = {}
dynamic_cat_cols = {}
dynamic_activity_col = {}
static_cat_cols = {}
dynamic_num_cols = {}
static_num_cols = {}
filename = {}

logs_dir = ""

#### Flood management log settings ####
dataset = "fmplog"

filename[dataset] = os.path.join(logs_dir, "FMPlog.csv")

case_id_col[dataset] = "Case ID"
activity_col[dataset] = "Activity"
timestamp_col[dataset] = "Complete Timestamp"
label_col[dataset] = "label"
neg_label[dataset] = "Adapt"
pos_label[dataset] = "noAdapt"

# features for classifier
dynamic_activity_col[dataset] = ["Activity"]
static_cat_cols[dataset] = ["Materiel Ressource","Human Ressource"]
static_num_cols[dataset] = []
dynamic_cat_cols[dataset] = ["risque level"] 
dynamic_num_cols[dataset] = ["water flow","water level"]

#### Traffic fines settings ####

#### Sepsis Cases settings ####
datasets = ["sepsis_cases_%s" % i for i in range(1, 5)]

for dataset in datasets:
    
    filename[dataset] = os.path.join(logs_dir, "%s.csv" % dataset)

    case_id_col[dataset] = "Case ID"
    activity_col[dataset] = "Activity"
    resource_col[dataset] = "org:group"
    timestamp_col[dataset] = "time:timestamp"
    label_col[dataset] = "label"
    pos_label[dataset] = "noAdapt"
    neg_label[dataset] = "Adapt"

    # features for classifier
    dynamic_activity_col[dataset] = ["Activity"]
    dynamic_cat_cols[dataset] = ["Activity", 'org:group'] # i.e. event attributes
    static_cat_cols[dataset] = ['Diagnose'] # i.e. case attributes that are known from the start
    dynamic_num_cols[dataset] = ['CRP', 'LacticAcid', 'Leucocytes']#, "hour", "weekday", "month", "timesincemidnight", "timesincelastevent", "timesincecasestart", "event_nr", "open_cases"]
    static_num_cols[dataset] = ['Age','DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG',
                       'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor',
                       'DiagnosticOther', 'DiagnosticSputum', 'DiagnosticUrinaryCulture',
                       'DiagnosticUrinarySediment', 'DiagnosticXthorax', 'DisfuncOrg',
                       'Hypotensie',    'SIRSCritHeartRate', 'SIRSCritTachypnea',
                       'SIRSCritTemperature']


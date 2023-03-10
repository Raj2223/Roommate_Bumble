import streamlit as st
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import hamming

warnings.filterwarnings('ignore')

#Import Data
original_data = pd.read_excel('find_my_buddy.csv.xlsx', sheet_name='Buddy Sheet')

original_data = original_data.drop(labels = [ 'i20 amount', 'VISA status', 'Planned VISA interview date', 'VISA consulate - city ',
           'Do you need a Flight mate?', 'If yes, then Flight Date?',
           'Any other comments  ?', 'Email address', 'Your Decision', 'Areas of Interest (In NEU )', 'Facebook profile link (optional)'], axis = 1)

original_data = original_data.drop(['Facilities required '], axis = 1)

original_data.columns = ['name','gender', 'hometown', 'current_city',
    'need_roommate', 'course',
    'edu_prog',
    'open_to_other_branch', 'undergrad_uni',
    'work_ex',
    'dist_from_uni',
    'person_per_room',
    'apt_type', 'rent_budget',
    'alcohol', 'smoking',
    'special_pref', 'food_pref',
    'cul_skills ', 'looking_for_roommate', 'hobbies']

original_data = original_data.drop(['edu_prog'], axis = 1)
original_data = original_data.drop(labels = ['special_pref'], axis = 1)
original_data = original_data.drop(labels=['undergrad_uni'], axis = 1)
original_data = original_data.drop(labels = ['hobbies'], axis = 1)
original_data = original_data.drop(labels = ['looking_for_roommate'], axis = 1)
original_data = original_data.rename(columns={'cul_skills ': 'cul_skills'})

#Data Cleaning
# st.write(original_data.columns)

def data_cleaning(original_data):
    # type(original_data)
    # ##print(original_data.columns)
    # original_data
    #Male, Female to 0, 1
    original_data['gender'].replace(['Male','Female'], [0, 1], inplace = True)
    # original_data.columns
    #Change names to more accessible ones

    original_data['rent_budget'] = original_data['rent_budget'].str.replace('$', '', regex = True)
    original_data['rent_budget'] = original_data['rent_budget'].str.replace('<', '', regex = True)
    # original_data['dist_from_uni'] = original_data['dist_from_uni'].str.replace('<', '', regex = True)
    #Start exploring each column and encoding them wherever needed

    #1) Course
    # original_data.course.unique()
    one_course = pd.get_dummies(original_data['course'])
    # Join one hot vectors of course and delete original
    original_data = original_data.join(one_course)

    original_data = original_data.drop('course', axis = 1)
    # Drop one vague reading
    
    # original_data
    
    #2) Hometown and current city
    original_data['hometown'] = original_data['hometown'].str.strip().str.lower()
    original_data['current_city'] = original_data['current_city'].str.strip().str.lower()
#     ##print('Proportion of people currently in the same city as their hometown:')
    
    # np.sum(original_data['hometown'] == original_data['current_city'])/len(original_data)
    
    
    # #3) Undergrad universities
   
    #4) Open to roommates from other branch
    pd.value_counts(original_data['open_to_other_branch'])
    # Fill NA's with the most frequent value
    original_data.open_to_other_branch = original_data.open_to_other_branch.fillna('Yes')
    # Check nulls now
    original_data.open_to_other_branch.isnull().sum()

    #5) Undergrad university
    # original_data.undergrad_uni.value_counts()
    
    
    original_data.isnull().sum()

    #6) Work experience
   
    original_data.work_ex.value_counts()
    # Get non null values, average them and change work ex to avg work ex value
    work_ex_avg = np.average(original_data.work_ex[original_data.work_ex.notnull()])

    original_data.work_ex = original_data.work_ex.fillna(work_ex_avg)
    original_data.work_ex.isnull().sum()

    #7) Distance from university
    # Get non null values, average them and change distance to avg distance value
    # Fix two vague values
    #original_data.set_value(206, 'dist_from_uni', '<10')
    original_data.at[206, 'dist_from_uni']='<10'
    #original_data.set_value(130, 'dist_from_uni', '<10')
    original_data.at[130, 'dist_from_uni']='<10'
    # Change distance to int
    # original_data.dist_from_uni[original_data.dist_from_uni.notnull()] = [int(d[1:]) for d in original_data.dist_from_uni[original_data.dist_from_uni.notnull()]]
    # original_data.dist_from_uni.isnull().sum()
    # # Get avg distance students prefer staying from university
    original_data['dist_from_uni'] = original_data['dist_from_uni'].astype(str).str.replace('<','', regex = True)
    original_data['dist_from_uni'] = original_data['dist_from_uni'].apply(float)
    dist_avg = np.sum(original_data.dist_from_uni[original_data.dist_from_uni.notnull()])/len(original_data.dist_from_uni\
                                                                                    [original_data.dist_from_uni.notnull()])
    # dist_avg

    # dist_avg = original_data['dist_from_uni']
    original_data.dist_from_uni = original_data.dist_from_uni.fillna(dist_avg)
    original_data.dist_from_uni.isnull().sum()
    original_data.isnull().sum()

    #8) Person Per Room
   
    hall_ind = 'I can stay in Hall too'
    hall_yes_no = [1 if hall_ind in str(d) else 0 for d in original_data.person_per_room]
    original_data['hall_yes_no'] = hall_yes_no
    # original_data
    
    pd.get_dummies(original_data.person_per_room)

    #Save this for further work.
    original_data.to_csv('original_data_2.csv')
   
    #Start new here with semi-cleaned CSV
    original_data = pd.read_csv('original_data_2.csv', index_col=0)
    # original_data
    original_data.isnull().sum()
    

    for ind, i in enumerate(original_data.person_per_room):
        if len(str(i).split(',')) == 1 and hall_ind in str(i):
            # ##print(ind, i)
            #original_data.set_value(ind, 'person_per_room', 2)
            original_data.at[ind, 'person_per_room']=2
    
    nansss = 0
    max_ppr = []
    for d in original_data.person_per_room:
        try:    
            if hall_ind in str(d):
                new_d = str(d).split(',')
                del(new_d[-1])
                max_per_room = (max(int(j) for j in new_d))

            else:
                max_per_room = (max(int(j) for j in str(d).split(',')))
            # ##print(max_per_room)
            max_ppr.append(max_per_room)

        except:
            nansss += 1
            max_ppr.append(d)
    #         ##print(d)
#     ##print('NAN COUNT:', nansss)
#     ##print('Length of ppr:', len(max_ppr))
    original_data['max_ppr'] = max_ppr

    original_data = original_data.drop(labels=['person_per_room'], axis = 1)

    # Now fill NA's with avg value of maximum number of people per room.
    original_data.max_ppr = original_data.max_ppr.fillna(np.average(original_data.max_ppr[original_data.max_ppr.notnull()]))
    original_data.isnull().sum()
    
    #9) Apartment type
    #Create dummy variable.

    original_data.apt_type = original_data.apt_type.fillna('Hall, 1BHK, 2 BHK, 3BHK, 4 BHK')
    apt_types = ['1 BHK', '2 BHK', '3 BHK', '4 BHK', 'Hall']
    for a in apt_types:
        original_data[a] = [0]*len(original_data)
    # original_data
    #Insert indicator values for apt type.

    for ind, apt_choice in enumerate(original_data.apt_type):
    # try:
        for at in apt_types:
            if at in str(apt_choice):
                #original_data.set_value(ind, at, 1)
                original_data.at[ind, at]=1

    #     except:
    #         ##print(apt_choice)
    original_data.isnull().sum()
    
    #10) Rent budget
    original_data.rent_budget.value_counts()
    import matplotlib.pyplot as plt
    x = original_data.rent_budget.value_counts()
    # original_data.rent_budget.hist()
    #Change budget to integer by taking just the numeric values.

    original_data.rent_budget = original_data.rent_budget.fillna('< 500')
    for ind, budget in enumerate(original_data.rent_budget):
        try:
            budget = str(budget).strip()
            original_data.set_value(ind, 'rent_budget', int(budget[3:]))
    #         break
        except:
            continue
    # original_data
    #Temporary function to check null values.

    def check_null():
        return(original_data.isnull().sum())
    original_data.rent_budget.value_counts()
    for ind, bud in enumerate(original_data.rent_budget):
    # try:
    #     ##print(bud)
        if '$1000 +' in str(bud):
                original_data.at[ind, 'rent_budget']=1000
    # check_null()
    
    
    #11) Alcohol
    # original_data.alcohol.value_counts()
    original_data.alcohol = original_data.alcohol.fillna('Flexible')

    original_data.alcohol = original_data.alcohol.replace('Flexible', 0)
    original_data.alcohol = original_data.alcohol.replace('Strictly NO', 1)
    original_data.alcohol = original_data.alcohol.replace('Strictly Yes', 2)
    # original_data.alcohol.value_counts()
    
    
    #12) Smoking
    # original_data.smoking.value_counts()
    original_data.smoking = original_data.smoking.fillna('Flexible')

    original_data.smoking = original_data.smoking.replace('Flexible', 0)
    original_data.smoking = original_data.smoking.replace('Strictly No', 1)
    original_data.smoking = original_data.smoking.replace('Strictly Yes', 2)
    original_data.smoking.value_counts()
    # check_null()
    #Special preferences does not add anything new of significance at least not uniformly. So dropping it.

    
    #13) Food preferences
    original_data.food_pref.value_counts()
    original_data.food_pref = original_data.food_pref.fillna('Flexible (I prefer  veg or Non -veg for myself but ready to live with anyone)')

    original_data.food_pref = original_data.food_pref.replace('Flexible (I prefer  veg or Non -veg for myself but ready to live with anyone)', 0)
    original_data.food_pref = original_data.food_pref.replace('Veg & Non-Veg', 0)
    original_data.food_pref = original_data.food_pref.replace('Strictly Veg', 1)
    original_data.food_pref = original_data.food_pref.replace('Strictly Non Veg', 2)
    # original_data.food_pref.value_counts()
    # check_null()
    
    
    #14) Cul skills
    # original_data.cul_skills.value_counts()
    original_data.cul_skills = original_data.cul_skills.fillna('Sometimes')

    original_data.cul_skills = original_data.cul_skills.replace('Sometimes', 0)
    original_data.cul_skills = original_data.cul_skills.replace('Expert', 1)
    original_data.cul_skills = original_data.cul_skills.replace('Never tried', 2)
    original_data.cul_skills.value_counts()
    check_null()
    original_data.looking_for_roommate.value_counts()
    original_data.looking_for_roommate = original_data.looking_for_roommate.fillna('Who can cook sometimes')

    original_data.looking_for_roommate = original_data.looking_for_roommate.replace('Who can cook sometimes', 0)
    original_data.looking_for_roommate = original_data.looking_for_roommate.replace('No culinary skills required', 1)
    check_null()
    # original_data
    original_data.to_csv('original_data_no_na.csv')
    #Drop apt_type since its an extra feature now, and hometown because we'll use just one location indicator(current_city) for now.

    original_data = original_data.drop(labels = ['apt_type', 'hometown'], axis = 1)
    
    
    #15) Current City
    original_data.current_city.value_counts()
    #Mumbai Bangalore and Pune are the 3 most frequent locations of Masters students in our dataset.

    #Changing cities to categorical.

    all_cities = original_data.current_city.unique()
    num_cities = list(range(len(all_cities)))

    city_num_dict = dict(zip(all_cities, num_cities))
    # city_num_dict
    original_data['current_city'] = original_data['current_city'].map(city_num_dict)
    
    
    #16) Open to other branch
    original_data.open_to_other_branch = original_data.open_to_other_branch.replace('Yes', 0)
    original_data.open_to_other_branch = original_data.open_to_other_branch.replace('No', 1)
    # original_data.isnull().sum()
    #Change variable names to make them more accessible.

    original_data = original_data.rename(columns = {'1BHK':'bhk_1', '2 BHK':'bhk_2', '3BHK':'bhk_3', '4 BHK':'bhk_4', 'Hall':'hall'})
    # original_data
    #original_data.columns
    #original_data.dtypes
    # original_data.rent_budget.value_counts()

    #This code uses the errors='coerce' argument to convert non-numeric values to NaN,
    # and then uses the replace() method to replace the NaN values with 0. 
    original_data.rent_budget = pd.to_numeric(original_data.rent_budget, errors='coerce')
    original_data.rent_budget = original_data.rent_budget.replace(np.nan, 0)
    original_data.rent_budget = pd.to_numeric(original_data.rent_budget)
    # original_data.rent_budget.value_counts()

     # original_data.dtypes
    #Save this cleaned data for future use
    original_data.to_csv('user_data_clean.csv')
#     ##print("Data clean Successfull")
    
    return original_data

# -----------------------------------------------------------------------user inputs--------------------------------------------------

# step-1 take user inputs

name_input = st.text_input('Name', value='Your Name' )

gender_input = st.radio(f"Gender",('Male', 'Female'),  horizontal = True )

current_city_input = {'mumbai': 0,
 'bangalore': 1,
 'kolkata': 2,
 'chennai': 3,
 'pune': 4,
 'hyderabad': 5,
 'nagpur': 6,
 'rajkot': 7,
 'wadala, mumbai': 8,
 'vadodara': 9,
 'kochi': 10,
 'raipur': 11,
 'goa': 12,
 'gurgaon': 13,
 'ann arbor, michigan': 14,
 'vijayawada': 15,
 'delhi': 16,
 'dehradun': 17,
 'surat': 18,
 'mysore': 19,
 'meerut': 20,
 'pune, maharashtra': 21,
 'bengaluru': 22,
 'navi mumbai': 23,
 'yelahanka': 24,
 'navi mumbai (panvel)': 25,
 'mangalore': 26,
 'jaipur': 27,
 'vellore': 28,
 'jalgaon': 29,
 'auranagabad': 30,
 'nellore': 31,
 'ahmedabad': 32,
 'sangli, mh': 33,
 'india': 34,
 'satara': 35,
 'thane': 36,
 'vapi': 37,
 'vidhya nagar': 38,
 'sharjah, uae': 39,
 'mumbai(bandra)': 40,
 'banaglore': 41,
 'bangalore, karnataka': 42,
 'faridabad': 43,
 'faridabad(ncr)': 44,
 'patras': 45,
 'miraj': 46,
 'bhopal': 47,
 'chandigarh': 48,
 'noida': 49,
 'coimbatore': 50,
 'indore': 51,
 'guntur': 52,
 'thane,mumbai': 53,
 'nigdi pune': 54,
 'new delhi': 55,
 'nit waranagal': 56,
 'dombivli': 57,
 'mumbai, india': 58,
 'dubai': 59,
 'kalyan': 60,
 'east windsor': 61,
 'borivali': 62,
 'tiruchirappalli': 63,
 'bhilai, durg': 64,
 'thane(mumbai)': 65,
 'bangalore, india': 66,
 'kerala': 67,
 'surendranagar': 68,
 'aurangabad': 69,
 'trichy': 70,
 'solapur': 71,
 'bhubaneswar': 72,
 'delhi ncr': 73,
 'us': 74,
 'hubli': 75,
 'chennaia': 76,
 'singapore': 77,
 'sangli': 78,
 'banglore': 79,
 'kota': 80,
 'jalandhar': 81,
 'boston': 82,
 'hosur': 83,
 'thanjavur': 84}

home_town_input = st.selectbox('Home Town',current_city_input)

currently_living_input = st.selectbox('Current City',current_city_input)

need_roommate_input = st.radio(f"Need Roommate?",('Yes', 'No'),  horizontal = True )

course_input = st.selectbox(f"Course",('MS Biotechnology','MS Civil Engineering','MS Computer Science','MS Computer Systems Engineering',
                             'MS Data Science','MS Electrical and Computer Engineering','MS Energy Systems','MS Engineering Management',
                             'MS Industrial Engineering','MS Information Assurance and Cyber Security','MS Information Systems',
                             'MS Mechanical Engineering','MS Project Management','Others'))


open_to_other_branch_input =  st.radio(f"Are you open to other branch room mate? Yes:1 or No:0",('1', '0'),  horizontal = True )

work_ex_input = st.number_input(label = 'Work Experience', min_value = 0)

dist_from_uni_input = st.number_input(label = 'Distance from University in kms')

person_per_room_input = st.multiselect(label = 'Persons in one room',options = ('1','2','3','4'))

apt_type_input = st.multiselect(f"Apartment Type",('1 BHK', '2 BHK', '3 BHK','4 BHK','Hall'))

rent_budget_input = st.number_input(label = 'Rent Budget', min_value = 500)

Alcohol_input =  st.radio(f"Alcohol",('Flexible', 'Strictly No'),  horizontal = True )

smoking_input =  st.radio(f"Smoking",('Flexible', 'Strictly No'),  horizontal = True )

Food_pref_input = st.radio(f"Food Preference",('Strictly Veg', 'Strictly Non Veg', 'Veg & Non-Veg'),  horizontal = True )

cul_skills_input = st.radio(f"Culinary skills",('Sometimes', 'Never tried', 'Expert'),  horizontal = True )

hobbies_input = st.text_input('Hobbies' )

new_data = {
'name' : name_input,
'gender': gender_input,
'hometown': home_town_input,
'current_city': currently_living_input,
'need_roommate': need_roommate_input,
'looking_for_roommate': need_roommate_input,
'course': course_input,
'open_to_other_branch': open_to_other_branch_input,
'work_ex': work_ex_input,
'dist_from_uni' : dist_from_uni_input,
'person_per_room': ', '.join(person_per_room_input),
'apt_type' : apt_type_input,
'rent_budget': rent_budget_input,
'alcohol': Alcohol_input,
'smoking' : smoking_input,
'food_pref': Food_pref_input,
'cul_skills' : cul_skills_input}

original_data = original_data.append(new_data, ignore_index = True)

#og_sheet = pd.read_csv("D:\\BlueThinQ\\Streamlit\\streamlit\\Roommate Bumble\\Roommate_Bumble-main\\find_my_buddy.csv", sep='|')


# step-2 add user inputs in the data (append variables in original data)
og_sheet = pd.read_excel('find_my_buddy.csv.xlsx', sheet_name='Buddy Sheet')
# og_sheet['Full Name'].drop(522)

# original_data = original_data.drop(index = [522]).reset_index(drop = True)
meta_data = data_cleaning(original_data)

# st.write(meta_data.shape[0])

meta_data = meta_data.drop(labels=['looking_for_roommate', 'Others'], axis = 1)
# name_list
# meta_data

# ##print("Enter Index No. to get Recommendation for that person ")
# ##print("Note: number should be between 1 to 520")

# x=int(input()) #-------- change it to latest index 

x = meta_data.shape[0] - 1 #-------- change it to latest index


name_list = list(og_sheet['Full Name'])
test_person = meta_data.iloc[[x]]  
# meta_data = meta_data.dropna()
# test_person

# st.write(test_person)

def get_cont_cat(dataframe, var_type):
   
    # Convert any series to dataframe
    # if not isinstance(dataframe, pd.DataFrame):
        # ##print('ip is not dataframe')
    cont_cols = ['work_ex', 'dist_from_uni', 'rent_budget']
    
    if var_type == 'cont':
        return dataframe[cont_cols]
    
    elif var_type == 'cat':
        return dataframe.drop(labels = cont_cols, axis = 1)
    
    else: raise ValueError('Variable type should be either "cont" or "cat"')

test_p_cont = np.array(get_cont_cat(test_person, 'cont'))
db_cont = np.array(get_cont_cat(meta_data, 'cont'))
#test_p_cont.shape
#db_cont.shape
#test_p_cont

#euclidean_distances(test_p_cont, db_cont).shape

def get_cont_dist(person, database, metric):

    to_std = np.vstack((person, database))
    
    all_std = StandardScaler().fit_transform(to_std)
    person_std = all_std[0,:].reshape(1,-1)
    database_std = all_std[1:,:]
    
    if metric == 'euclidean':
        cont_distance_matrix = euclidean_distances(person_std, database_std)
        return cont_distance_matrix

#hamming_distances

def get_cat_dist(person, database, metric):
    
    cat_distance_matrix = []
    if metric == 'hamming':
        database_df = pd.DataFrame(database)
        for index, c_row in database_df.iterrows():
            cat_distance_matrix.append(hamming(person, c_row))
    return(np.array(cat_distance_matrix)) 

test_cat = get_cont_cat(test_person, 'cat')
database_cat = get_cont_cat(meta_data[meta_data['gender'] == 0], 'cat')
# get_cat_dist(test_cat.to_numpy().ravel(), database_cat.to_numpy(), 'hamming')[57]


test_cat_array = get_cont_cat(test_person, 'cat').to_numpy().ravel()
database_cat_array = get_cont_cat(meta_data[meta_data['gender'] == 0], 'cat').to_numpy()
# get_cat_dist(test_cat_array, database_cat_array, 'hamming')[57]

test_cat_array = get_cont_cat(test_person, 'cat').to_numpy().ravel()
database_cat_array = get_cont_cat(meta_data, 'cat').to_numpy()
# get_cat_dist(test_cat_array, database_cat_array, 'hamming')

test_cat = get_cont_cat(test_person, 'cat').to_numpy().ravel()
test_cat2 = get_cont_cat(meta_data.iloc[[x]], 'cat').to_numpy().ravel()

hamming(test_cat, test_cat2)

def findRoommate(new_person, database, n_roommates, alpha, beta):
    # Split data by gender to reduce computations
    database_g = database[database['gender'] == new_person.iloc[0]['gender']]
    name_g = [name_list[i] for i in list(database_g.index) if i < len(name_list)]

    # Split new datapoint into continuous and categorical sets
    new_person_cont = get_cont_cat(new_person, 'cont').to_numpy().flatten()
    new_person_cat = get_cont_cat(new_person, 'cat').to_numpy().flatten()

    # Split database into continuous and categorical sets
    database_cont = get_cont_cat(database_g, 'cont').to_numpy()
    database_cat = get_cont_cat(database_g, 'cat').to_numpy()

    # Get distances for both continuous and categorical sets
    dist_cont = get_cont_dist(new_person_cont, database_cont, 'euclidean')
    dist_cat = get_cat_dist(new_person_cat, database_cat, 'hamming')

    # Create final distance matrix of weighted average
    final_dist = alpha*dist_cont + beta*dist_cat

    # Sort the distance matrix to get top n roommates
    top_n_matches = np.argsort(final_dist)[0][1 : n_roommates + 1]
    
    
    top_n_dict = {"index": top_n_matches.tolist(),
                  "name": [name_g[j] for j in top_n_matches]}

    # Print the top n matches in index:name format
    for i in range(len(top_n_matches)):
        print(f'{top_n_matches[i]}:{name_g[top_n_matches[i]]}')
        
    #print Details
    print("\n")
    print(test_person,"\n\n")
    
    for i in range(len(top_n_matches)):
        print(top_n_matches[i])
        print(meta_data.iloc[[i]],"\n")
    
    return top_n_dict

l = findRoommate(test_person, meta_data, 5, 1, 1)

output_index = l['index']
current_index = x
new = original_data.iloc[current_index]

# button to display user  data
if st.button('Your data'):
    st.table(new)

# button to display recommended roommates data
# Display top 5 recommended roommates
if st.button('Get Recommendation'):
    st.write(original_data.iloc[output_index].drop(labels = ['looking_for_roommate'], axis =1))

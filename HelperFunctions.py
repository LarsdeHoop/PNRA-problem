import pickle
import copy
import datetime

debug = False
def dprint(string, end = "\n"):
    """Only prints if global debug variable is True"""
    global debug
    if debug:
        print(string, end = end)

def set_debug(value):
    """Set the global debug variable to another value"""
    global debug
    debug = value

def avg_of_list(data_list):
    if type(data_list[0]) == datetime.timedelta:
        timedelta_in_seconds = [t.total_seconds() for t in data_list]
        avg_timedelta_seconds = sum(timedelta_in_seconds) / len(timedelta_in_seconds)
        return datetime.timedelta(seconds=avg_timedelta_seconds)
    else:
        return sum(data_list) / len(data_list)

def shift_day_to_index(day, shift_types, shift_type="early"):
    return 3 * day + shift_types.index(shift_type)

def shift_index_to_day(shift_types, shift_index):
    day = shift_index // 3
    shift_type = shift_types[shift_index % 3]
    return day, shift_type

def copy_solution(solution):
    """Copy a solution such that changes do not affect other solutions"""
    solution_copy = (copy.deepcopy(solution[0]), copy.deepcopy(solution[1]))
    return solution_copy

def pickle_store(data, filename, filepath):
    """Store a data object in a pickle file"""
    with open(filepath + "/" + filename, 'wb') as f:
        pickle.dump(data, f)
    dprint(f"Successfully stored data in {filename}")

def pickle_load(filename, filepath):
    """Retrieve a data object from a pickle file"""
    with open(filepath + "/" + filename, 'rb') as f:
        data = pickle.load(f)
    dprint(f"Successfully loaded data from {filename}")
    return data

def instance_name_to_dataset(instance_name):
    """
    Determine the directory name corresponding to the instance name.
    If this is to be run on other machines or for other instances, change accordingly.
    """
    if instance_name[0] == "t":
        if "_" in instance_name:
            return "emergency_test"
        elif "-" not in instance_name:
            return "ihtc2024_test_combined"
        else:
            return "ihtc2024_test_short"
    elif instance_name[0] == "i":
        if "_" in instance_name:
            return "emergency_public"
        else:
            return "ihtc2024_public_combined"
    elif instance_name[0] == "m":
        if "_" in instance_name:
            return "emergency_hidden"
        else:
            return "ihtc2024_hidden_combined"
    else:
        return "Other"

def get_instance_size(instance):
    if len(instance.patient_ids) < 75:
        return "small"
    elif len(instance.patient_ids) < 240:
        return "medium"
    else:
        return "large"

def print_as_table(column_header, columns, lines_with_hline = [], bolded_indices = [],
                   align = "^", sep = " | ", end = "", print_hline = False):
    """
    Function to print data in a table format for legibility or for easy copying into latex.
    Parameters
        - column_names: list of column names as strings
        - columns: list of lists containing the values/strings in each column
        - align: optional aligning parameter, could be one of [^,>,<] or
                 could be a list of these with alignment for each column
        - sep: separator between each column or
               could be a list as a seperator for between each column
        - end: An optional end of each row (for example \\ for latex tables)
        - print_hline: whether to print a line of dashes below the table headers
    """


    if column_header is None:
        column_names = ["" for _ in range(len(columns))]
    else:
        column_names = column_header

    number_of_columns = len(column_names)
    if number_of_columns != len(columns):
        raise Exception("Number of columns does not match number of columns.")

    column_lengths = [len(columns[i]) for i in range(number_of_columns)]
    if min(column_lengths) != max(column_lengths):
        raise Exception(f"Column lengths are not the same: {column_lengths}.")
    column_length = max(column_lengths)

    # Turn items into strings if needed
    for i in range(number_of_columns):
        for j in range(column_length):
            if type(columns[i][j]) != str:
                columns[i][j] = str(columns[i][j])

    # Add bolding if specified
    for (row,col) in bolded_indices:
        columns[col][row] = "\\textbf{" + columns[col][row] + "}"

    column_widths = []
    for i in range(number_of_columns):
        header_width = len(column_names[i])
        values_width = max([len(x) for x in columns[i]])
        max_width = max(header_width,values_width)
        column_widths.append(max_width)


    if type(align) == str:
        alignments = [align for _ in range(number_of_columns)]
    else:
        alignments = align

    if type(sep) == str:
        separators = [sep for _ in range(number_of_columns)]
    else:
        separators = sep

    total_width = sum(column_widths) + sum([len(sep) for sep in separators[:-1]]) + 2

    # print header
    print()
    header = ""
    for col in range(number_of_columns):
        header += f"{column_names[col]:^{column_widths[col]}}"
        if col != number_of_columns - 1:
            header += separators[col]
    header += end
    if column_header is not None:
        print(header)

    # print(sep.join([f"{column_names[i]:^{column_widths[i]}}" for i in range(number_of_columns)]) + end)

    # print horizontal line below header
    if print_hline:
        print("=" * total_width)

    # print other values
    for row in range(column_length):
        # print each row
        line = ""
        for col in range(number_of_columns):
            line += f"{columns[col][row]:{alignments[col]}{column_widths[col]}}"
            if col != number_of_columns - 1:
                line += separators[col]
        line += end

        if row in lines_with_hline:
            line += r" \hline"
        print(line)
    return column_widths

def get_solution_attributes(instance, solution):
    """
    Get additional data structures to compute the objective or to compute the change in objective when making
    a local search move. Will return a dictionary with the following keys:
        - patients_per_room: a dictionary with for each room-shift pair a list of patients staying in that room
                             during that shift.
        - nurse_per_room: a dictionary with for each room-shift pair, the nurse assigned to that room.
        - gender_numb_per_room: a dictionary with for each room-shift-gender tuple the number of patients of
                                that gender staying in that room during that shift.
        - gender_assignment: a dictionary with for each room-early_shift pair the gender of the room on that day.
                             Value can be either "A", "B", "Both" or None.
        - NP_assignment: a dictionary with for each patient-shift pair, the nurse assigned to that patient
                         during that shift.
        - workload_per_room: a dictionary with for each room-shift pair, the assigned workload of that room
                             during that shift.
        - workload_per_nurse: a dictionary with for each nurse-shift pair, the assigned workload of that nurse
                              during that shift.
        - nurse_workload_order: a dictionary with for each shift, the sorted workload of the nurses present.
        - nurse_count_per_patient: a dictionary with for each patient, a dictionary counting during how
                                   many shifts a nurse was assigned to them
        - least_assigned_nurse_per_patient: a dictionary with for each patient, a tuple of a nurse id and the number of times
                                 this nurse was assigned to the patient. This is the least assigned nurse.
    """
    PR_assignment, NR_assignment = solution

    # note that patients_per_rooms contains the same patients for the early, late and night shift,
    # but this makes it easier in the computation later
    patients_per_room = {(r, s): [] for r in instance.room_ids for s in instance.all_shifts}
    for p in instance.patient_ids:
        assigned_room = PR_assignment[p]
        for s in instance.patients[p]["shifts"]:
            patients_per_room[assigned_room, s].append(p)

    nurse_per_room = dict()
    for n in instance.nurse_ids:
        for s in instance.nurses[n]["shifts"]:
            for r in NR_assignment[n,s]:
                if (r, s) not in nurse_per_room:
                    nurse_per_room[r, s] = n
                else:
                    raise Exception(
                        f"{(r, s)} was already in nurse_per_room with value {nurse_per_room[r, s]} before assigning {n}")

    gender_numb_per_room = {(r, s, gender): 0 for r in instance.room_ids for s in instance.early_shifts
                            for gender in ["A", "B"]}
    for p in instance.patient_ids:
        assigned_room = PR_assignment[p]
        gender = instance.patients[p]["gender"]
        for s in instance.patients[p]["shifts"]:
            if s in instance.early_shifts:
                gender_numb_per_room[assigned_room, s, gender] += 1

    gender_assignment = dict()
    for r in instance.room_ids:
        for s in instance.early_shifts:
            if gender_numb_per_room[r,s,"A"] == 0 and gender_numb_per_room[r,s,"B"] == 0:
                gender_assignment[r, s] = None
            elif gender_numb_per_room[r,s,"A"] == 0 and gender_numb_per_room[r,s,"B"] > 0:
                gender_assignment[r, s] = "B"
            elif gender_numb_per_room[r,s,"A"] > 0 and gender_numb_per_room[r,s,"B"] == 0:
                gender_assignment[r, s] = "A"
            else:
                gender_assignment[r, s] = "Both"

    NP_assignment = dict()
    for p in instance.patient_ids:
        assigned_room = PR_assignment[p]
        for s in instance.patients[p]["shifts"]:
            nurse_in_room = nurse_per_room[assigned_room, s]
            NP_assignment[p, s] = nurse_in_room

    workload_per_room = {(r,s):0 for r in instance.room_ids for s in instance.all_shifts}
    for p in instance.patients:
        assigned_room = PR_assignment[p]
        for s in instance.patients[p]["shifts"]:
            workload_per_room[assigned_room,s] += instance.patients[p]["workload"][s]

    workload_per_nurse = {(n, s): 0 for n in instance.nurse_ids for s in instance.nurses[n]["shifts"]}
    for n in instance.nurse_ids:
        for s in instance.nurses[n]["shifts"]:
            for r in NR_assignment[n,s]:
                workload_per_nurse[n, s] += workload_per_room[r,s]

    nurse_workload_order = dict()
    for s in instance.all_shifts:
        nurse_workload_per_shift = []
        for n in instance.nurses_per_shift[s]:
            rel_workload = workload_per_nurse[n, s] / instance.nurses[n]["max_load"][s]
            nurse_workload_per_shift.append((n,rel_workload))

        sorted_nurse_workload_per_shift = sorted(nurse_workload_per_shift, key = lambda x: x[1])
        nurse_workload_order[s] = sorted_nurse_workload_per_shift

    nurse_count_per_patient = {p: dict() for p in instance.patient_ids}
    least_assigned_nurse_per_patient = dict()
    for p in instance.patient_ids:
        assigned_room = PR_assignment[p]
        for s in instance.patients[p]["shifts"]:
            assigned_nurse = nurse_per_room[assigned_room, s]
            if assigned_nurse not in nurse_count_per_patient[p]:
                nurse_count_per_patient[p][assigned_nurse] = 1
            else:
                nurse_count_per_patient[p][assigned_nurse] += 1

        min_val = float('inf')
        min_nurse = None
        for n in nurse_count_per_patient[p].keys():
            if nurse_count_per_patient[p][n] < min_val:
                min_val = nurse_count_per_patient[p][n]
                min_nurse = n
        least_assigned_nurse_per_patient[p] = (min_nurse, min_val)

    # Store datastructures in dictionary
    solution_attributes = dict()
    solution_attributes["patients_per_room"] = patients_per_room
    solution_attributes["nurse_per_room"] = nurse_per_room
    solution_attributes["gender_numb_per_room"] = gender_numb_per_room
    solution_attributes["gender_assignment"] = gender_assignment
    solution_attributes["NP_assignment"] = NP_assignment
    solution_attributes["workload_per_room"] = workload_per_room
    solution_attributes["workload_per_nurse"] = workload_per_nurse
    solution_attributes["nurse_workload_order"] = nurse_workload_order
    solution_attributes["nurse_count_per_patient"] = nurse_count_per_patient
    solution_attributes["least_assigned_nurse_per_patient"] = least_assigned_nurse_per_patient

    return solution_attributes

def get_lower_upper_bounds():
    lower_bounds = {'test01': 223.1, 'test02': 304.05, 'test03': 137.333, 'test04': 308.717, 'test05': 164.983,
                    'test06': 375.017, 'test07': 724.0, 'test08': 482.283, 'test09': 600.267, 'test10': 1967.25,
                    'i01': 146.167, 'i02': 240.533, 'i03': 156.492, 'i04': 446.717, 'i05': 339.933, 'i06': 279.1,
                    'i07': 461.95, 'i08': 1056.533, 'i09': 542.983, 'i10': 717.5, 'i11': 383.767, 'i12': 652.067,
                    'i13': 794.083, 'i14': 652.1, 'i15': 684.75, 'i16': 902.317, 'i17': 1509.717, 'i18': 731.017,
                    'i19': 1201.95, 'i20': 811.7, 'i21': 1332.45, 'i22': 2038.133, 'i23': 1456.65, 'i24': 1647.117,
                    'i25': 1032.351, 'i26': 2040.7, 'i27': 2276.433, 'i28': 1085.3, 'i29': 1380.717, 'i30': 1703.818}
    upper_bounds = {'test01': 1072.1, 'test02': 1452.217, 'test03': 519.0, 'test04': 2003.7, 'test05': 778.967,
                    'test06': 2257.283, 'test07': 3968.217, 'test08': 2999.917, 'test09': 5904.717, 'test10': 26102.8,
                    'i01': 537.3, 'i02': 1056.183, 'i03': 510.983, 'i04': 2605.717, 'i05': 2457.8, 'i06': 1504.117,
                    'i07': 3040.383, 'i08': 8782.967, 'i09': 3957.533, 'i10': 3796.967, 'i11': 2605.7, 'i12': 4039.05,
                    'i13': 4368.667, 'i14': 5206.2, 'i15': 5535.917, 'i16': 5973.267, 'i17': 12027.083, 'i18': 6329.0,
                    'i19': 12168.35, 'i20': 5888.933, 'i21': 10669.7, 'i22': 14815.6, 'i23': 14750.1, 'i24': 16938.5,
                    'i25': 9897.633, 'i26': 17591.1, 'i27': 18404.9, 'i28': 10941.5, 'i29': 11326.3, 'i30': 14398.5}
    return lower_bounds, upper_bounds




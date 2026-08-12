from HelperFunctions import *
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def create_PR_assignment(instance, sorting_order = "-", default_value = 0.0):
    # Initialize dictionaries
    PR_assignment = dict()  # p->r
    room_occupancy = {(r, s): 0 for r in instance.room_ids for s in instance.early_shifts}
    room_gender = {(r, s): "Empty" for r in instance.room_ids for s in instance.early_shifts}

    # Assign current occupants to the correct room and update room_occupancy & room_gender
    dprint("Assigning current occupants to the correct rooms:")
    for p in instance.occupant_ids:
        prev_room = instance.patients[p]['prev_room']
        PR_assignment[p] = prev_room
        for s in instance.patients[p]["shifts"]:
            if s in instance.early_shifts:
                room_occupancy[prev_room, s] += 1
                if room_gender[prev_room, s] == "Empty":
                    room_gender[prev_room, s] = instance.patients[p]["gender"]
                elif room_gender[prev_room, s] != instance.patients[p]["gender"]:
                    room_gender[prev_room, s] = "Both"

    # Order unassigned patients chronologically based on admission date and ascendingly in terms of LOS
    other_patients = [p for p in instance.non_occupant_ids]  # make a copy
    other_patients = sorted(other_patients, key=lambda p: (instance.patients[p]["shifts"][0],
                                                           instance.patients[p]["shifts"][-1]))

    # Create the similarity matrix
    if "S" in sorting_order:
        sim_matrix = create_similarity_matrix(instance)
    else:
        sim_matrix = dict()

    # Store room_information and sorting_information to reduce arguments of PR_recursive
    room_information = PR_assignment, room_occupancy, room_gender
    sorting_information = sorting_order, default_value, sim_matrix

    # Start algorithm
    start_time = datetime.datetime.now()
    feasible_assignment, room_information = PR_recursive(0, other_patients, room_information,
                                                         sorting_information, instance, start_time)

    if not feasible_assignment:
        print("Found no feasible assignment")
        return None

    PR_assignment = room_information[0]
    return PR_assignment

def PR_recursive(patient_index, patients, room_information, sorting_information, instance, start_time):
    if (datetime.datetime.now() - start_time).total_seconds() > 600: # incorporate a time limit of ten minutes
        return False, room_information

    if patient_index == len(patients):
        return True, room_information

    # get the room and sorting information
    PR_assignment, room_occupancy, room_gender = room_information
    sorting_order, default_value, sim_matrix = sorting_information

    # get patient attributes
    p = patients[patient_index]
    adm_shift = instance.patients[p]["shifts"][0]
    LOS = len(instance.patients[p]["shifts"]) // 3
    gender = instance.patients[p]["gender"]
    incomp_rooms = instance.patients[p]['incompatible_rooms']
    dprint(f"\nNow assigning patient {p}: "
           f"adm shift {adm_shift} & "
           f"LOS {LOS} & "
           f"gender {gender} & "
           f"invalid rooms {incomp_rooms}")

    # define the opposite gender
    if gender == "A":
        opp_gender = "B"
    else:
        opp_gender = "A"

    # determine which rooms are feasible
    feasible_rooms = []
    for r in instance.room_ids:
        # Check if the room is incompatible for the patient
        if r in incomp_rooms:
            dprint(f"\tRoom {r} is incompatible")
            continue

        # Check if the room is fully occupied (since we schedule chronologically, only check admission shift)
        if room_occupancy[r, adm_shift] == instance.room_capacities[r]:
            dprint(f"\tRoom {r} is already at capacity ({instance.room_capacities[r]})")
            continue

        feasible_rooms.append(r)
    dprint(f"Feasible rooms: {feasible_rooms}")

    # Calculate the similarity score per room
    room_sim_dict = room_sim_dict = {r:0 for r in instance.room_ids}
    if "S" in sorting_order:
        # determine the previously assigned patients that overlap with current patient
        numb_patients_per_room = {r: 0 for r in instance.room_ids}
        patient_shifts = set(instance.patients[p]["shifts"])
        for p0 in instance.occupant_ids + patients[:patient_index]:
            if len(patient_shifts & set(instance.patients[p0]["shifts"])) != 0:
                r = PR_assignment[p0]
                numb_patients_per_room[r] += 1
                room_sim_dict[r] += sim_matrix[p,p0]

                if r == "r14" and p == "p047":
                    print(f"patient {p0} adds {sim_matrix[p,p0]}")

        # take the average per room
        for r in instance.room_ids:
            if numb_patients_per_room[r] != 0:
                room_sim_dict[r] /= numb_patients_per_room[r]
            else:
                room_sim_dict[r] = default_value


    #####
    # Sort the feasible rooms based on the given criterion
    criterion_map = {
        'G': lambda r: sum(room_gender[r, adm_shift + 3 * i] == opp_gender for i in range(LOS)),
        'O': lambda r: sum(room_occupancy[r, adm_shift + 3 * i] for i in range(LOS)),
        'R': lambda r: -sum([instance.room_capacities[r] - room_occupancy[r, adm_shift + 3 * i] for i in range(LOS)]),
        'S': lambda r: -room_sim_dict[r]
    }

    if sorting_order != "-":
        feasible_rooms = sorted(feasible_rooms, key=lambda r: tuple(criterion_map[c](r) for c in sorting_order))


    # Select room and continue
    for r in feasible_rooms:
        dprint(f"Assigning patient {p} to room {r}")
        PR_assignment[p] = r

        # copy room information
        room_occupancy_copy = room_occupancy.copy()
        room_gender_copy = room_gender.copy()

        # update dictionaries
        for s in instance.patients[p]["shifts"]:
            if s in instance.early_shifts:
                room_occupancy_copy[r,s] += 1
                if room_gender_copy[r,s] == "Empty":
                    room_gender_copy[r,s] = gender
                elif room_gender_copy[r,s] != gender and room_gender_copy[r,s] != "Both":
                    room_gender_copy[r,s] = "Both"

        # Recurse
        room_information_copy = PR_assignment, room_occupancy_copy, room_gender_copy
        feasible_assignment, room_information_temp = PR_recursive(patient_index + 1, patients, room_information_copy,
                                                                  sorting_information, instance, start_time)

        if feasible_assignment:
            return True, room_information_temp

    if len(feasible_rooms) == 0:
        dprint("No rooms feasible")
    else:
        dprint("No room yielded a feasible assignment")
    return False, room_information

def create_similarity_matrix(instance):
    """Give each patient pair a similarity score based on how similar the CoC nurse assignment is.
    Takes the best continuity of care value from 5 set covers per patient"""

    #####################
    # RUN SET COVER ILP #
    #####################

    nurse_options_per_patient = dict()
    for p in instance.patient_ids:
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        model = gp.Model(f"Continuity lower bound", env=env)
        model.Params.LogToConsole = False
        model.Params.Seed = 42
        model.Params.PoolSearchMode = 2
        model.Params.PoolSolutions = 5
        model.Params.PoolGap = 0.0

        ever_assigned = model.addVars(instance.nurse_ids, lb=0.0, ub=1.0, vtype=GRB.BINARY,
                                      name="ever_assigned")

        model.addConstrs(gp.quicksum(ever_assigned[n] for n in instance.nurse_ids
                                     if s in instance.nurses[n]["shifts"]) >= 1
                         for s in instance.patients[p]["shifts"])

        model.setObjective(gp.quicksum(ever_assigned[n] for n in instance.nurse_ids),
                           GRB.MINIMIZE)

        model.optimize()

        options = []
        for sol in range(model.SolCount):
            model.Params.SolutionNumber = sol
            nurses_per_patient = []
            for n in instance.nurse_ids:
                if abs(ever_assigned[n].PoolNX - 1) < 10e-6:
                    nurses_per_patient.append(n)
            options.append(nurses_per_patient)
        nurse_options_per_patient[p] = options

    #####################
    # SIMILARITY MATRIX #
    #####################

    similarity_matrix = dict()
    for i in range(len(instance.patient_ids) - 1):
        p1 = instance.patient_ids[i]
        p1_shifts = set(instance.patients[p1]["shifts"])
        for j in range(i + 1, len(instance.patient_ids)):
            p2 = instance.patient_ids[j]
            p2_shifts = set(instance.patients[p2]["shifts"])
            best_sim_score = 0
            if len(p1_shifts & p2_shifts) != 0:
                for p1_nurses_list in nurse_options_per_patient[p1]:
                    for p2_nurses_list in nurse_options_per_patient[p2]:
                        p1_nurses = set(p1_nurses_list)
                        p2_nurses = set(p2_nurses_list)

                        union = p1_nurses | p2_nurses
                        intersection = p1_nurses & p2_nurses
                        # dprint(f"Union -> {union}")
                        dprint(f"intersection -> {intersection}")

                        sim_score = len(intersection) / (min(len(p1_nurses), len(p2_nurses)))
                        # sim_score = len(intersection) / len(union)
                        dprint(f"sim_score {p1, p2} = {sim_score:.3f}\n")
                        if sim_score > best_sim_score:
                            best_sim_score = sim_score

            similarity_matrix[p1, p2] = best_sim_score
            similarity_matrix[p2, p1] = best_sim_score
    return similarity_matrix

def show_PR_assignment(instance, PR_assignment):
    # Show it in one figure
    gender_colors = {"A": "tab:blue", "B": "tab:orange", "Both": "tab:green", "Empty": "white"}
    gender_names = {"A": "Male", "B": "Female", "Empty": "Empty", "Both": "Both"}
    patches = [mpatches.Rectangle((0, 0), width=0, height=0, facecolor=v,
                                  label=gender_names[k], linewidth=1, edgecolor="black")
               for k, v in gender_colors.items()]

    box_height = 1
    y_tick_fontsize = 10
    if len(instance.room_ids) > 15:
        figsize = [9.6, 9.6]
        margin_val = 0.01
    else:
        figsize = [6.4, 4.8]
        margin_val = 0.02

    fig, ax = plt.subplots(constrained_layout=True, figsize=figsize)

    # Define the margins beforehand as this is used in the fontsize calculation
    plt.xlim(-3 * margin_val * instance.numberOfDays, 3 * (1 + margin_val) * instance.numberOfDays)
    plt.ylim(-margin_val * len(instance.room_ids) - 0.5, (1 + margin_val) * len(instance.room_ids) - 0.5)

    fontsize = max(8, min(14, 100 // len(instance.room_ids)))
    # print(instance.instance_name, fontsize, text_margin)

    gender_assignment = {(r,s):"Empty" for r in instance.room_ids for s in instance.early_shifts}
    numb_patients_per_room = {(r,s):0 for r in instance.room_ids for s in instance.early_shifts}
    for p in PR_assignment:
        r = PR_assignment[p]
        gender = instance.patients[p]["gender"]
        for s in instance.patients[p]["shifts"]:
            if s in instance.early_shifts:
                numb_patients_per_room[r,s] += 1
                if gender_assignment[r,s] == "Empty":
                    gender_assignment[r, s] = gender
                elif gender_assignment[r,s] != gender:
                    gender_assignment[r, s] = "Both"


    for r in instance.room_ids:
        for s in instance.early_shifts:
            ax.barh(instance.room_ids.index(r), width=3, left=s, height=box_height,
                    color=gender_colors[gender_assignment[r, s]], linewidth=1,
                    edgecolor="black")
            if numb_patients_per_room[r, s] < instance.room_capacities[r]:
                color='white'
            else:
                color = "red"
            ax.text(s + 1.5, instance.room_ids.index(r), str(numb_patients_per_room[r, s]), ha="center",
                    color=color, va="center_baseline", weight="bold", fontsize=fontsize)


    plt.xticks([1.5 + 3 * i for i in range(len(instance.early_shifts))],
               [str(i + 1) for i in range(len(instance.early_shifts))])

    plt.yticks(list(range(len(instance.room_ids))),
               [f"{r} ({instance.room_capacities[r]})" for r in instance.room_ids],
               fontsize=y_tick_fontsize)

    plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(f"Gender and occupancy for instance {instance.instance_name}")
    plt.xlabel("Day")
    plt.ylabel("Rooms")
    plt.show()


if __name__ == "__main__":
    from InstanceClass import Instance
    instance_name = "m19"
    dataset_name = instance_name_to_dataset(instance_name)
    file_path = "Instances/" + dataset_name + "/" + instance_name + ".json"
    instance = Instance(file_path, print_instance_info=False)

    start_time = datetime.datetime.now()
    PR_assignment = create_PR_assignment(instance, sorting_order = "GSO", default_value=1.0)
    end_time = datetime.datetime.now()
    print(f"PR-assignment took {(end_time - start_time).total_seconds()} seconds")
    show_PR_assignment(instance, PR_assignment)

    # multiple["i"+str(i).zfill(2) for i in range(1,31)]
    # for instance_name in ["i"+str(i).zfill(2) for i in range(21,31)]:
    #     dataset_name = instance_name_to_dataset(instance_name)
    #     file_path = "Instances/" + dataset_name + "/" + instance_name + ".json"
    #     instance = Instance(file_path, print_instance_info=False)
    #
    #     start_time = datetime.datetime.now()
    #     PR_assignment = create_PR_assignment(instance, sorting_order = "GSO", default_value=0.8)
    #     end_time = datetime.datetime.now()
    #     print(f"Took {(end_time - start_time).total_seconds()} seconds")
    #     show_PR_assignment(instance, PR_assignment)
    #
    #     start_time = datetime.datetime.now()
    #     PR_assignment = create_PR_assignment(instance, sorting_order = "GSO", default_value=1.0)
    #     end_time = datetime.datetime.now()
    #     print(f"Took {(end_time - start_time).total_seconds()} seconds")
    #     show_PR_assignment(instance, PR_assignment)

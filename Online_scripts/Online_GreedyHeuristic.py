import random
from HelperFunctions import *
import gurobipy as gp
from gurobipy import GRB

def get_greedy_solution(instance, use_sequential = True, use_warm_start = False):

    # initialize PR_assignment and NR_assignment
    solution_history = dict() # s_update -> solution
    fixed_PR_assignment = dict()  # (p, s) -> r
    fixed_NR_assignment = dict()

    # get the parameters for each update shift
    params_per_update_shift = get_params_per_update_shift(instance)
    norel_fracs_per_update_shift = get_norel_fracs_per_update_shift(instance)


    # Create the similarity matrix
    sim_matrix = dict()
    for s_update in params_per_update_shift:
        if "S" in params_per_update_shift[s_update][0]:
            sim_matrix = create_similarity_matrix(instance)
            break

    # define the time limit based on the number of remaining days
    numb_remaining_days_per_s_update = {s_update:instance.numberOfDays - s_update // 3 for s_update in instance.schedule_update_shifts}
    time_limit_per_s_update = {s_update: 20 * numb_remaining_days_per_s_update[s_update] for s_update in instance.schedule_update_shifts}

    # Run the greedy algorithm for every update shift
    warm_start_values = dict()
    for i, cur_shift in enumerate(instance.schedule_update_shifts):
        # print(f"CURRENT SHIFT {cur_shift}")

        # get the correct parameters
        sorting_order, default_value, shift_order, sign = params_per_update_shift[cur_shift]

        temp_PR_assignment = create_PR_assignment(instance, fixed_PR_assignment, cur_shift, sim_matrix,
                                              sorting_order=sorting_order, default_value=default_value)

        if cur_shift == 0 or not use_warm_start:
            warm_start = False
        else:
            warm_start = True

        if use_sequential:
            temp_NR_assignment, warm_start_values = create_NR_assignment_sequential(instance, cur_shift, fixed_NR_assignment,
                                                        temp_PR_assignment, warm_start_values, warm_start,
                                                        shift_order=shift_order, sign=sign)
        else:
            time_limit = time_limit_per_s_update[cur_shift]
            norel_time = time_limit * norel_fracs_per_update_shift[cur_shift]
            temp_NR_assignment, warm_start_values = create_NR_assignment_non_sequential(instance, cur_shift,
                                                        fixed_NR_assignment, temp_PR_assignment, warm_start_values,
                                                        warm_start, time_limit = time_limit, noRel_heur_time=norel_time,
                                                        print_ilp_log = False)

        solution_history[cur_shift] = (temp_PR_assignment, temp_NR_assignment)

        # Fix some values
        if i == len(instance.schedule_update_shifts) - 1:
            fixed_PR_assignment = temp_PR_assignment
            for (n,s) in temp_NR_assignment:
                fixed_NR_assignment[n,s] = temp_NR_assignment[n,s]
        else:
            next_shift = instance.schedule_update_shifts[i + 1]
            for s in range(cur_shift, next_shift, 3): # only early shifts
                for p in instance.patients_per_shift[s]:
                    fixed_PR_assignment[p, s] = temp_PR_assignment[p,s]
            for s in range(cur_shift, next_shift):
                for n in instance.nurses_per_shift[s]:
                    fixed_NR_assignment[n,s] = temp_NR_assignment[n,s]

    final_solution = (fixed_PR_assignment, fixed_NR_assignment)
    return final_solution, solution_history

def create_PR_assignment(instance, fixed_PR_assignment, current_shift,
                         sim_matrix, sorting_order = "-", default_value = 0.0):
    # Initialize dictionaries
    PR_assignment = fixed_PR_assignment.copy()  # (p,s) -> r
    room_occupancy = {(r, s): 0 for r in instance.room_ids for s in instance.early_shifts}
    room_gender = {(r, s): "Empty" for r in instance.room_ids for s in instance.early_shifts}

    # Assign previous patients to correct room and determine current occupants
    occupants_in_current_shift = instance.occupants_per_update_shift[current_shift]
    prev_room_per_occupant = dict()
    dprint("Determining current occupants and fixing values:")

    for p in occupants_in_current_shift:
        if current_shift == 0:
            prev_room = instance.patients[p]["prev_room"]
            prev_room_per_occupant[p] = prev_room
            dprint(f"patient {p} in {prev_room}")
        else:
            prev_room = fixed_PR_assignment[p, current_shift - 3]
            prev_room_per_occupant[p] = prev_room
            dprint(f"patient {p} in {prev_room}")

    # print(occupants_in_current_shift)
    # print(prev_room_per_occupant)

    # Determine what patients can be scheduled at the current instance
    patients_to_schedule = []
    for p in instance.schedulable_patients[current_shift]:
        s_dep = instance.patients[p]["shifts"][-1]
        if s_dep >= current_shift:
            # only if they are not gone yet
            patients_to_schedule.append(p)

    # sort the patients
    patients_to_schedule = sorted(patients_to_schedule, key=lambda p: (-(p in occupants_in_current_shift),
                                                                       instance.patients[p]["shifts"][0],
                                                                       instance.patients[p]["shifts"][-1]))

    # Store room_information and sorting_information to reduce arguments of PR_recursive
    room_information = PR_assignment, room_occupancy, room_gender
    occupant_information = occupants_in_current_shift, prev_room_per_occupant
    sorting_information = sorting_order, default_value, sim_matrix

    # Start algorithm
    start_time = datetime.datetime.now()
    feasible_assignment, room_information = PR_recursive(0, patients_to_schedule, current_shift, instance,
                                             room_information, occupant_information, sorting_information, start_time)

    if not feasible_assignment:
        raise Exception(f"Found no feasible assignment for instance {instance.instance_name}")

    PR_assignment = room_information[0]

    return PR_assignment

def PR_recursive(patient_index, patients, current_shift, instance,
                 room_information, occupant_information, sorting_information, start_time):
    if (datetime.datetime.now() - start_time).total_seconds() > 10: # incorporate a time limit of ten seconds
        return False, room_information

    if patient_index == len(patients):
        return True, room_information

    # get the room and sorting information
    PR_assignment, room_occupancy, room_gender = room_information
    occupants_in_current_shift, prev_room_per_occupant = occupant_information
    sorting_order, default_value, sim_matrix = sorting_information

    # get patient attributes
    p = patients[patient_index]
    LOS = len(instance.patients[p]["shifts"]) // 3
    gender = instance.patients[p]["gender"]
    incomp_rooms = instance.patients[p]['incompatible_rooms']
    is_occupant = p in occupants_in_current_shift
    if is_occupant:
        shifts_to_consider = [s for s in instance.patients[p]["shifts"] if s >= current_shift]
    else:
        shifts_to_consider = instance.patients[p]["shifts"]
    dprint(f"\nNow assigning patient {p}: "
           f"shifts to consider {shifts_to_consider} & "
           f"LOS {LOS} & "
           f"gender {gender} & "
           f"invalid rooms {incomp_rooms} & {is_occupant}")

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

        # Check if the room is fully occupied
        at_capacity = False
        for s in shifts_to_consider:
            if s in instance.early_shifts:
                if room_occupancy[r, s] == instance.room_capacities[r]:
                    dprint(f"\tRoom {r} is already at capacity ({instance.room_capacities[r]})")
                    at_capacity = True
                    continue

        if not at_capacity:
            feasible_rooms.append(r)
    dprint(f"Feasible rooms: {feasible_rooms}")

    # Calculate the similarity score per room
    room_sim_dict = room_sim_dict = {r:0 for r in instance.room_ids}
    if "S" in sorting_order:
        # determine the previously assigned patients that overlap with current patient
        numb_patients_per_room = {r: 0 for r in instance.room_ids}
        patient_shifts = set(shifts_to_consider)
        for p0 in patients[:patient_index]:
            p0_shifts = instance.patients[p0]["shifts"]
            if len(patient_shifts & set(p0_shifts)) != 0:
                # we look at this shift because afterwards p0 will not transfer
                s_to_look_at = max(p0_shifts[0], shifts_to_consider[0])
                r = PR_assignment[p0, s_to_look_at]
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
        'G': lambda r: sum(room_gender[r, s] == opp_gender for s in shifts_to_consider if s in instance.early_shifts),
        'O': lambda r: sum(room_occupancy[r, s] for s in shifts_to_consider if s in instance.early_shifts),
        'R': lambda r: r != prev_room_per_occupant.get(p, None),
        'S': lambda r: -room_sim_dict[r],
        "I": lambda r: sum([r in instance.patients[p]["incompatible_rooms"] for p in instance.patient_ids])
    }


    if sorting_order != "-":
        feasible_rooms = sorted(feasible_rooms, key=lambda r: tuple(criterion_map[c](r) for c in sorting_order))

    # print(f"patient {p}: rooms {feasible_rooms}\n")

    # Select room and continue
    for r in feasible_rooms:
        dprint(f"Assigning patient {p} to room {r}")
        for s in shifts_to_consider:
            if s in instance.early_shifts:
                PR_assignment[p,s] = r

        # copy room information
        room_occupancy_copy = room_occupancy.copy()
        room_gender_copy = room_gender.copy()

        # update dictionaries
        for s in instance.patients[p]["shifts"]:
            if s in instance.early_shifts:
                room_occupancy_copy[r, s] += 1
                if room_gender_copy[r, s] == "Empty":
                    room_gender_copy[r, s] = gender
                elif room_gender_copy[r, s] != gender and room_gender_copy[r, s] != "Both":
                    room_gender_copy[r, s] = "Both"

        # Recurse
        room_information_copy = PR_assignment, room_occupancy_copy, room_gender_copy
        feasible_assignment, room_information_temp = PR_recursive(patient_index + 1, patients, current_shift, instance,
                                                                  room_information_copy, occupant_information,
                                                                  sorting_information, start_time)

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

def create_NR_assignment_sequential(instance, cur_shift, fixed_NR_assignment, PR_assignment,
                                    warm_start_values, use_warm_start = False,
                                    shift_order = "chrono", sign=1, print_ilp_log = False):

    NR_assignment = {(n,s):[] for s in instance.all_shifts
                           for n in instance.nurses_per_shift[s] if s >= cur_shift}
    for (n,s) in fixed_NR_assignment:
        NR_assignment[n,s] = fixed_NR_assignment[n,s]

    # set nurses_per_patient based on the fixed NR_assignment values
    nurses_per_patient = {p: [] for p in instance.schedulable_patients[cur_shift]}
    for p in instance.occupants_per_update_shift[cur_shift]:
        for s in instance.patients[p]["shifts"]:
            if s < cur_shift: # go over things that are fixed
                if s in instance.early_shifts:
                    r = PR_assignment[p, s]
                elif s in instance.late_shifts:
                    r = PR_assignment[p, s - 1]
                else:
                    r = PR_assignment[p, s - 2]

                # find which nurse is assigned to room r
                for n in instance.nurses_per_shift[s]:
                    if r in fixed_NR_assignment[n, s]:
                        if n not in nurses_per_patient[p]:
                            nurses_per_patient[p].append(n)

    # Sort the shifts to schedule according to the
    all_shifts_to_schedule = [s for s in instance.all_shifts if s >= cur_shift]
    if shift_order == "chrono":
        all_shifts_to_schedule = sorted(all_shifts_to_schedule, key=lambda s: sign * s)
    elif shift_order == "nurses":
        all_shifts_to_schedule = sorted(all_shifts_to_schedule, key=lambda s: sign * len(instance.nurses_per_shift[s]))
    elif shift_order == "patients":
        all_shifts_to_schedule = sorted(all_shifts_to_schedule, key=lambda s: sign * len(instance.patients_per_shift[s]))
    elif shift_order == "workload":
        workload_per_shift = {s: sum([instance.patients[p]["workload"][s] for p in instance.patients_per_shift[s]])
                              for s in instance.all_shifts}
        all_shifts_to_schedule = sorted(all_shifts_to_schedule, key=lambda s: sign * workload_per_shift[s])
    elif shift_order == "avg_workload":
        workload_per_shift = {s: sum([instance.patients[p]["workload"][s] for p in instance.patients_per_shift[s]])
                              for s in instance.all_shifts}
        numb_patients_per_shift = {s: len(instance.patients_per_shift[s]) for s in instance.all_shifts}
        for s in instance.all_shifts:
            if numb_patients_per_shift[s] == 0:
                numb_patients_per_shift[s] = 1  # to prevent div by zero error
        all_shifts_to_schedule = sorted(all_shifts_to_schedule,
                                      key=lambda s: sign * (workload_per_shift[s] / numb_patients_per_shift[s]))
    else:
        random.shuffle(all_shifts_to_schedule)

    # solve an ILP for each shift
    for s in all_shifts_to_schedule:
        nurses_present = instance.nurses_per_shift[s]
        patients_present = instance.known_patients_per_shift[s, cur_shift]

        env = gp.Env(empty=True)
        if not print_ilp_log:
            env.setParam('OutputFlag', 0)
        env.start()

        model = gp.Model(f"PR-assignment {instance.instance_name} - shift {s}", env=env)

        if not print_ilp_log:
            model.Params.LogToConsole = False
        model.Params.TimeLimit = 30
        # model.Params.Seed = random.randint(1, 100000)
        model.Params.Seed = 42
        # unlike the regular greedy heuristic, we run this heuristic only once per instance
        # the randomness will therefore not be tested in this way
        # because we will most likely run the size classes on separate occasions,
        # we just set a single seed for replicability

        ## VARIABLES
        # Variable for nurse-to-room assignment
        valid_x_indices = [(n, r) for n in nurses_present for r in instance.room_ids]
        x = model.addVars(valid_x_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="x")
        if use_warm_start:
            for (n,r) in valid_x_indices:
                x[n,r].Start = warm_start_values[n,r,s]

        # Variable for nurse-to-patient assignment
        valid_z_indices = [(n, p) for n in nurses_present for p in patients_present]
        z = model.addVars(valid_z_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="z")

        ## CONSTRAINTS
        # Each room assigned to exactly one nurse
        model.addConstrs(gp.quicksum(x[n, r] for n in nurses_present) == 1 for r in instance.room_ids)

        # NPA is the according to PR_assignment and NRA
        if s in instance.early_shifts:
            model.addConstrs(z[n, p] == x[n, PR_assignment[p,s]] for n in nurses_present for p in patients_present)
        elif s in instance.late_shifts:
            model.addConstrs(z[n, p] == x[n, PR_assignment[p,s-1]] for n in nurses_present for p in patients_present)
        else:
            model.addConstrs(z[n, p] == x[n, PR_assignment[p,s-2]] for n in nurses_present for p in patients_present)

        ## OBJECTIVE
        obj = gp.LinExpr()

        ## Minimizing number of different nurses
        if instance.weights["Continuity"] != 0:
            ever_assigned = model.addVars(nurses_present, patients_present, lb=0.0, ub=1.0, vtype=GRB.BINARY,
                                          name="ever_assigned")

            # set the previously assigned nurses to 1, otherwise depends on z_np
            for p in patients_present:
                for n in nurses_present:
                    if n in nurses_per_patient[p]:
                        model.addConstr(ever_assigned[n, p] == 1)
                    else:
                        model.addConstr(ever_assigned[n, p] == z[n, p])

            # Add to objective
            obj += instance.weights["Continuity"] * gp.quicksum(
                ever_assigned[n, p] for n in nurses_present for p in patients_present)

        ## Skill requirement
        if instance.weights["Skill Requirements"] != 0:
            skill_vio = model.addVars(patients_present, lb=0.0, vtype=GRB.CONTINUOUS)

            model.addConstrs(skill_vio[p] >= instance.patients[p]["skill"][s] -
                             gp.quicksum(instance.nurses[n]["skill"] * z[n, p]
                                         for n in nurses_present)
                             for p in patients_present)

            obj += instance.weights["Skill Requirements"] * gp.quicksum(skill_vio[p] for p in patients_present)

        ## Minimizing workload violation
        if instance.weights["Workload Violation"] != 0:
            load_vio = model.addVars(nurses_present, lb=0.0, vtype=GRB.CONTINUOUS, name="load_vio")

            model.addConstrs(gp.quicksum(instance.patients[p]["workload"][s] * z[n, p]
                                         for p in patients_present)
                             <= instance.nurses[n]["max_load"][s] + load_vio[n]
                             for n in nurses_present)

            # Add to objective
            obj += instance.weights["Workload Violation"] * gp.quicksum(load_vio[n] for n in nurses_present)

        # Minimizing workload imbalance per shift
        if instance.weights["Workload Imbalance"] != 0:
            min_load = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="min_load")
            max_load = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="max_load")

            model.addConstrs(min_load <= gp.quicksum(instance.patients[p]["workload"][s]
                                                     / instance.nurses[n]["max_load"][s] * z[n, p]
                                                     for p in patients_present)
                             for n in nurses_present)
            model.addConstrs(max_load >= gp.quicksum(instance.patients[p]["workload"][s]
                                                     / instance.nurses[n]["max_load"][s] * z[n, p]
                                                     for p in patients_present)
                             for n in nurses_present)

            # Add to objective
            obj += instance.weights["Workload Imbalance"] * (max_load - min_load)

        model.setObjective(obj, GRB.MINIMIZE)
        model.optimize()


        # Update NR assignment
        for n in nurses_present:
            for r in instance.room_ids:
                if abs(x[n, r].X - 1) < 10e-6:  # possible rounding error
                    NR_assignment[n, s].append(r)

        # update nurses_per_patient
        if instance.weights["Continuity"] != 0:
            for p in patients_present:
                for n in nurses_present:
                    if abs(ever_assigned[n, p].X - 1) < 10e-6:
                        if n not in nurses_per_patient[p]:
                            nurses_per_patient[p].append(n)

        # update warm_start values
        for (n,r) in valid_x_indices:
            warm_start_values[n,r,s] = x[n,r].X

    return NR_assignment, warm_start_values

def create_NR_assignment_non_sequential(instance, cur_shift, fixed_NR_assignment, PR_assignment,
                                        warm_start_values, use_warm_start = False,
                                        time_limit = 300, noRel_heur_time = 0, print_ilp_log = False):

    NR_assignment = {(n,s):[] for s in instance.all_shifts
                           for n in instance.nurses_per_shift[s] if s >= cur_shift}
    for (n,s) in fixed_NR_assignment:
        NR_assignment[n,s] = fixed_NR_assignment[n,s]

    # ILP must take into account previous nurse assignments
    nurses_per_patient = {p: [] for p in instance.schedulable_patients[cur_shift]}
    for p in instance.occupants_per_update_shift[cur_shift]:
        for s in instance.patients[p]["shifts"]:
            if s < cur_shift: # go over things that are fixed
                if s in instance.early_shifts:
                    r = PR_assignment[p, s]
                elif s in instance.late_shifts:
                    r = PR_assignment[p, s - 1]
                else:
                    r = PR_assignment[p, s - 2]

                # find which nurse is assigned to room r
                for n in instance.nurses_per_shift[s]:
                    if r in fixed_NR_assignment[n, s]:
                        if n not in nurses_per_patient[p]:
                            nurses_per_patient[p].append(n)

    # determine which shifts, patients and nurses are actually used in the ILP
    relevant_shifts = [s for s in instance.all_shifts if s >= cur_shift]
    relevant_patients = instance.schedulable_patients[cur_shift]
    relevant_nurses = []
    for n in instance.nurse_ids:
        last_shift = instance.nurses[n]["shifts"][-1]
        if last_shift >= cur_shift:
            relevant_nurses.append(n)


    env = gp.Env(empty=True)
    if not print_ilp_log:
        env.setParam('OutputFlag', 0)
    env.start()
    model = gp.Model(f"NR-assignment {instance.instance_name}", env=env)

    if not print_ilp_log:
        model.Params.LogToConsole = False

    model.Params.TimeLimit = time_limit
    model.Params.NoRelHeurTime = noRel_heur_time

    # Variable for nurse-to-room assignment (only shifts after current shift)
    valid_x_indices = [(n, r, s) for s in relevant_shifts for r in instance.room_ids
                       for n in instance.nurses_per_shift[s]]
    x = model.addVars(valid_x_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="x")
    if use_warm_start:
        for (n, r, s) in valid_x_indices:
            x[n, r, s].Start = warm_start_values[n, r, s]


    # Variable for nurse-to-patient assignment (only shifts after current shift)
    valid_z_indices = [(n, p, s) for s in relevant_shifts for p in instance.known_patients_per_shift[s,cur_shift]
                       for n in instance.nurses_per_shift[s]]
    z = model.addVars(valid_z_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="z")

    ## CONSTRAINTS
    # Each room assigned to exactly one nurse
    model.addConstrs(gp.quicksum(x[n, r, s] for n in instance.nurses_per_shift[s]) == 1
                     for r in instance.room_ids for s in relevant_shifts)

    # Set z_nps = 1 if nurse n assigned to correct room during shift s
    # due to annoying storing of PR_assignment, this constraint is quite ugly
    model.addConstrs(z[n, p, s] == x[n, PR_assignment[p, s], s] for s in instance.early_shifts if s >= cur_shift
                     for n in instance.nurses_per_shift[s] for p in instance.known_patients_per_shift[s, cur_shift])
    model.addConstrs(z[n, p, s] == x[n, PR_assignment[p, s - 1], s] for s in instance.late_shifts if s >= cur_shift
                     for n in instance.nurses_per_shift[s] for p in instance.known_patients_per_shift[s, cur_shift])
    model.addConstrs(z[n, p, s] == x[n, PR_assignment[p, s - 2], s] for s in instance.night_shifts if s >= cur_shift
                     for n in instance.nurses_per_shift[s] for p in instance.known_patients_per_shift[s, cur_shift])

    obj = gp.LinExpr()

    ## Continuity of care
    if instance.weights["Continuity"] != 0:
        ever_assigned = model.addVars(relevant_nurses, relevant_patients,
                                      lb=0.0, ub=1.0, vtype=GRB.BINARY, name="ever_assigned")


        for n in relevant_nurses:
            for p in relevant_patients:
                if n in nurses_per_patient[p]:
                    model.addConstr(ever_assigned[n, p] == 1)
                else:
                    model.addConstrs(ever_assigned[n,p] >= z[n,p,s] for s in relevant_shifts
                                     if s in instance.nurses[n]["shifts"]
                                     if s in instance.patients[p]["shifts"])
                    model.addConstr(ever_assigned[n, p] <= gp.quicksum(z[n, p, s] for s in relevant_shifts
                                                            if s in instance.patients[p]["shifts"]
                                                            if s in instance.nurses[n]["shifts"]))

        # Add to objective
        obj += (instance.weights["Continuity"] *
                gp.quicksum(ever_assigned[n, p] for n in relevant_nurses for p in relevant_patients))

    ## Skill requirement
    if instance.weights["Skill Requirements"] != 0:
        valid_indices = [(p, s) for s in relevant_shifts
                         for p in instance.known_patients_per_shift[s, cur_shift]]
        skill_vio = model.addVars(valid_indices, lb=0.0, vtype=GRB.CONTINUOUS)

        model.addConstrs(skill_vio[p, s] >= instance.patients[p]["skill"][s] -
                         gp.quicksum(instance.nurses[n]["skill"] * z[n, p, s]
                                     for n in instance.nurses_per_shift[s])
                         for s in relevant_shifts
                         for p in instance.known_patients_per_shift[s, cur_shift])

        obj += instance.weights["Skill Requirements"] * gp.quicksum(
            skill_vio[p, s]  for s in relevant_shifts
                         for p in instance.known_patients_per_shift[s, cur_shift])

    ## Minimizing workload violation
    if instance.weights["Workload Violation"] != 0:
        valid_indices = [(n, s) for s in relevant_shifts
                         for n in instance.nurses_per_shift[s]]
        load_vio = model.addVars(valid_indices, lb=0.0, vtype=GRB.CONTINUOUS, name="load_vio")

        model.addConstrs(gp.quicksum(instance.patients[p]["workload"][s] * z[n, p, s]
                                     for p in instance.known_patients_per_shift[s, cur_shift])
                         <= instance.nurses[n]["max_load"][s] + load_vio[n, s]
                         for s in relevant_shifts
                         for n in instance.nurses_per_shift[s])

        # Add to objective
        obj += instance.weights["Workload Violation"] * gp.quicksum(
            load_vio[n, s] for s in relevant_shifts
                         for n in instance.nurses_per_shift[s])

    ## Minimizing workload imbalance per shift
    if instance.weights["Workload Imbalance"] != 0:
        min_load = model.addVars(relevant_shifts, lb=0.0, vtype=GRB.CONTINUOUS, name="min_load")
        max_load = model.addVars(relevant_shifts, lb=0.0, vtype=GRB.CONTINUOUS, name="max_load")

        model.addConstrs(min_load[s] <= gp.quicksum(instance.patients[p]["workload"][s]
                                                    / instance.nurses[n]["max_load"][s] * z[n, p, s]
                                                    for p in instance.known_patients_per_shift[s, cur_shift])
                         for s in relevant_shifts for n in instance.nurses_per_shift[s])
        model.addConstrs(max_load[s] >= gp.quicksum(instance.patients[p]["workload"][s]
                                                    / instance.nurses[n]["max_load"][s] * z[n, p, s]
                                                    for p in instance.known_patients_per_shift[s, cur_shift])
                         for s in relevant_shifts for n in instance.nurses_per_shift[s])

        # Add to objective
        obj += instance.weights["Workload Imbalance"] * gp.quicksum(
            max_load[s] - min_load[s] for s in relevant_shifts)

    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.Status == GRB.Status.TIME_LIMIT:
        print(f"\tTime limit reached. Suboptimal solution (obj {model.ObjVal:.3f}, bound {model.ObjBound:.3f}, gap {100*model.MIPGap:.2f}%)")

    for s in relevant_shifts:
        for n in instance.nurses_per_shift[s]:
            for r in instance.room_ids:
                if abs(x[n, r, s].X - 1) < 10e-6:  # possible rounding error
                    NR_assignment[n, s].append(r)

    # update the warm start values
    for (n,r,s) in valid_x_indices:
        warm_start_values[n,r,s] = x[n,r,s].X

    return NR_assignment, warm_start_values

def get_params_per_update_shift(instance):
    params_per_update_shift = {}
    for s_update in instance.schedule_update_shifts:
        numb_patients_to_schedule = len(instance.schedulable_patients[s_update])
        if numb_patients_to_schedule < 75:
            sorting_order = "RGSO"
            default_value = 1
            shift_order = "patients"
            sign = -1
        elif numb_patients_to_schedule < 240:
            sorting_order = "RGSO"
            default_value = 0.8
            shift_order = "chrono"
            sign = 1
        else:
            sorting_order = "RGSO"
            default_value = 0.8
            shift_order = "chrono"
            sign = 1
        params_per_update_shift[s_update] = (sorting_order, default_value, shift_order, sign)

    return params_per_update_shift

def get_norel_fracs_per_update_shift(instance):
    norel_fracs_per_update_shift = {}
    for s_update in instance.schedule_update_shifts:
        numb_patients_to_schedule = len(instance.schedulable_patients[s_update])
        if numb_patients_to_schedule < 75:
            norel_frac = 0
        elif numb_patients_to_schedule < 240:
            norel_frac = 0.5
        else:
            norel_frac = 1
        norel_fracs_per_update_shift[s_update] = norel_frac

    return norel_fracs_per_update_shift


if __name__ == "__main__":
    from Online_InstanceClass import EmergencyInstance
    from Online_ComputeObjective import compute_objective_emergency

    instance_name = "m01_10_1"
    dataset_name = instance_name_to_dataset(instance_name)
    file_path = "Instances/" + dataset_name + "/" + instance_name + ".json"
    instance = EmergencyInstance(file_path, print_instance_info=False)

    use_sequential = True # whether to use the sequential or non-sequential solution
    use_warm_start = True # whether to provide the ILPs with the previous solution as warm start

    start_time = datetime.datetime.now()
    solution, solution_history = get_greedy_solution(instance, use_sequential, use_warm_start)
    end_time = datetime.datetime.now()
    print(f"Greedy heuristic took {(end_time - start_time).total_seconds()} seconds")
    compute_objective_emergency(instance, solution, print_table=True)

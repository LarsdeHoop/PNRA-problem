from InstanceClass import Instance
from HelperFunctions import *
import datetime
import gurobipy as gp
from gurobipy import GRB
import re

def ILP_formulation(instance, solve_it = True, time_limit = None, norel_heur_time = 0,
                    relax_problem = False, print_ilp_log = False, store_ilp_log = False, 
                    logfile_folder = "./logfiles/"):
    starting_time = datetime.datetime.now()
    dprint(f"Starting at {starting_time}")

    env = gp.Env(empty=True)
    if not (store_ilp_log or print_ilp_log):
        env.setParam('OutputFlag', 0)
    env.start()

    model = gp.Model(instance.instance_name, env=env)
    model.Params.NoRelHeurTime = norel_heur_time
    # model.Params.Method = 2

    if not print_ilp_log:
        model.Params.LogToConsole = False

    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    #################
    #   VARIABLES   #
    #################

    # decision variable for nurses
    valid_x_indices = [(n, r, s) for n in instance.nurse_ids for r in instance.room_ids
                       for s in instance.nurses[n]["shifts"]]
    x = model.addVars(valid_x_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="x")

    # decision variable for patients
    valid_y_indices = [(p, r) for p in instance.patient_ids for r in instance.room_ids]
    y = model.addVars(valid_y_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="y")

    # Variable for nurse-to-patient assignment
    valid_z_indices = [(n, p, s) for n in instance.nurse_ids for p in instance.patient_ids
                       for s in instance.nurses[n]["shifts"] if s in instance.patients[p]["shifts"]]
    z = model.addVars(valid_z_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="z")

    #################
    #  Constraints  #
    #################

    ## Nurse-to-Room assignment
    # Each room assigned to exactly one nurse
    model.addConstrs(gp.quicksum(x[n, r, s] for n in instance.nurses_per_shift[s]) == 1
                     for r in instance.room_ids for s in instance.all_shifts)

    ## Patient-to-Room assignment
    # Each patient is only assigned to one room
    model.addConstrs(gp.quicksum(y[p, r] for r in instance.room_ids) == 1 for p in instance.patient_ids)

    # Each room cannot exceed its capacity
    model.addConstrs(gp.quicksum(y[p, r] for p in instance.patients_per_shift[s])
                     <= instance.room_capacities[r] for r in instance.room_ids for s in instance.early_shifts)

    # Patients cannot be assigned to incompatible rooms
    model.addConstrs(y[p, r] == 0 for p in instance.patient_ids for r in instance.patients[p]["incompatible_rooms"])

    # Keep current occupants in the same room as before
    model.addConstrs(y[o, instance.patients[o]["prev_room"]] == 1 for o in instance.occupant_ids)

    ## Nurse-to-Patient assignment
    # Nurse is assigned to a patient if they are in the same room
    model.addConstrs(z[n, p, s] >= x[n, r, s] + y[p, r] - 1 for s in instance.all_shifts
                     for r in instance.room_ids for n in instance.nurses_per_shift[s]
                     for p in instance.patients_per_shift[s])

    # Only one nurse assigned to each patient in each shift
    model.addConstrs(gp.quicksum(z[n, p, s] for n in instance.nurses_per_shift[s])
                     == 1 for s in instance.all_shifts for p in instance.patients_per_shift[s])

    #################
    #   OBJECTIVE   #
    #################

    obj = gp.LinExpr()

    ## Minimizing number of different nurses
    if instance.weights["Continuity"] != 0:
        ever_assigned = model.addVars(instance.nurse_ids, instance.patient_ids, lb=0.0, ub=1.0, vtype=GRB.BINARY,
                                      name="ever_assigned")

        # ever_assigned = 1 if at least one time nurse n assigned to patient p
        model.addConstrs(ever_assigned[n, p] >= z[n, p, s] for s in instance.all_shifts
                         for n in instance.nurses_per_shift[s]
                         for p in instance.patients_per_shift[s])

        # If z = 0 for all shifts, then ever_assigned = 0
        model.addConstrs(ever_assigned[n, p] <= gp.quicksum(z[n, p, s] for s in instance.all_shifts
                                                            if s in instance.patients[p]["shifts"]
                                                            if s in instance.nurses[n]["shifts"])
                         for n in instance.nurse_ids for p in instance.patient_ids)

        # Add to objective
        obj += instance.weights["Continuity"] * gp.quicksum(ever_assigned[n, p] for n in instance.nurse_ids for p in instance.patient_ids)

    ## Minimizing number of gender violations
    if instance.weights["Gender-Mixing"] != 0:
        # Variable for gender mixing constraint
        f_in_room = model.addVars(instance.room_ids, instance.early_shifts, lb=0.0, ub=1.0, vtype=GRB.BINARY,
                                  name="f_in_room")
        m_in_room = model.addVars(instance.room_ids, instance.early_shifts, lb=0.0, ub=1.0, vtype=GRB.BINARY,
                                  name="m_in_room")
        gender_vio = model.addVars(instance.room_ids, instance.early_shifts, lb=0.0, ub=1.0, vtype=GRB.BINARY,
                                   name="gender_vio")

        # set m_in_room and f_in_room correctly based on assigned patients
        model.addConstrs(y[p, r] <= m_in_room[r, s] for s in instance.early_shifts for p in instance.patients_per_shift[s]
                         for r in instance.room_ids if instance.patients[p]["gender"] == "A")
        model.addConstrs(y[p, r] <= f_in_room[r, s] for s in instance.early_shifts for p in instance.patients_per_shift[s]
                         for r in instance.room_ids if instance.patients[p]["gender"] == "B")

        # If both male and female patients, add a violation
        model.addConstrs(m_in_room[r, s] + f_in_room[r, s] <= 1 + gender_vio[r, s] for r in instance.room_ids
                         for s in instance.early_shifts)

        obj += instance.weights["Gender-Mixing"] * gp.quicksum(gender_vio[r, s] for r in instance.room_ids for s in instance.early_shifts)

    ## Skill requirement
    if instance.weights["Skill Requirements"] != 0:
        valid_indices = [(p, s) for p in instance.patient_ids for s in instance.patients[p]["shifts"]]
        skill_vio = model.addVars(valid_indices, lb=0.0, vtype=GRB.CONTINUOUS)

        model.addConstrs(skill_vio[p, s] >= instance.patients[p]["skill"][s] -
                         gp.quicksum(instance.nurses[n]["skill"] * z[n, p, s]
                                     for n in instance.nurses_per_shift[s])
                         for p in instance.patient_ids for s in instance.patients[p]["shifts"])

        obj += instance.weights["Skill Requirements"] * gp.quicksum(
            skill_vio[p, s] for p in instance.patient_ids for s in instance.patients[p]["shifts"])

    ## Minimizing workload violation
    if instance.weights["Workload Violation"] != 0:
        valid_indices = [(n, s) for n in instance.nurse_ids for s in instance.nurses[n]["shifts"]]
        load_vio = model.addVars(valid_indices, lb=0.0, vtype=GRB.CONTINUOUS, name="load_vio")

        model.addConstrs(gp.quicksum(instance.patients[p]["workload"][s] * z[n, p, s]
                                     for p in instance.patients_per_shift[s])
                         <= instance.nurses[n]["max_load"][s] + load_vio[n, s]
                         for n in instance.nurse_ids for s in instance.nurses[n]["shifts"])

        # Add to objective
        obj += instance.weights["Workload Violation"] * gp.quicksum(
            load_vio[n, s] for n in instance.nurse_ids for s in instance.nurses[n]["shifts"])

    # Minimizing workload imbalance per shift
    if instance.weights["Workload Imbalance"] != 0:
        min_load = model.addVars(instance.all_shifts, lb=0.0, vtype=GRB.CONTINUOUS, name="min_load")
        max_load = model.addVars(instance.all_shifts, lb=0.0, vtype=GRB.CONTINUOUS, name="max_load")

        model.addConstrs(min_load[s] <= gp.quicksum(instance.patients[p]["workload"][s]
                                                    / instance.nurses[n]["max_load"][s] * z[n, p, s]
                                                    for p in instance.patients_per_shift[s])
                         for n in instance.nurse_ids for s in instance.nurses[n]["shifts"])
        model.addConstrs(max_load[s] >= gp.quicksum(instance.patients[p]["workload"][s]
                                                    / instance.nurses[n]["max_load"][s] * z[n, p, s]
                                                    for p in instance.patients_per_shift[s])
                         for n in instance.nurse_ids for s in instance.nurses[n]["shifts"])

        # Add to objective
        obj += instance.weights["Workload Imbalance"] * gp.quicksum(max_load[s] - min_load[s] for s in instance.all_shifts)

    model.setObjective(obj, GRB.MINIMIZE)

    # Retrieve model information
    model_created_time = datetime.datetime.now()
    creation_time = model_created_time - starting_time
    model.update()
    numb_vars_bin = len([1 for x in model.getVars() if x.VType == GRB.BINARY])
    numb_vars_cont = len([1 for x in model.getVars() if x.VType == GRB.CONTINUOUS])
    numb_constr = len(model.getConstrs())

    if print_ilp_log:
        print(f"Model {model.ModelName} defined at {model_created_time}, which took {creation_time}")
        print(f"Model contains {numb_vars_bin + numb_vars_cont} variables ({numb_vars_bin} binary and {numb_vars_cont} "
              f"continuous) and {numb_constr} constraints")

    # in case we want to relax the problem
    if relax_problem:
        # model.Params.Method = 2
        for v in model.getVars():
            v.setAttr('Vtype', 'CONTINUOUS')

    if solve_it:
        if store_ilp_log:
            callback = DefaultCallback(logfile_folder + instance.instance_name)
            optimize_start = datetime.datetime.now()
            model.optimize(callback)
            callback.logfile.close()
            logfile = callback.logfile_name
        else:
            optimize_start = datetime.datetime.now()
            model.optimize()
            logfile = None

        optimizing_finished = datetime.datetime.now()
        solving_time = optimizing_finished - optimize_start
        dprint(f"Finished optimizing at {optimizing_finished}. Solving took {solving_time}")

        obj_value = model.ObjVal
        best_bound = model.ObjBound
        if model.Status == 2 or model.Status == 15:
            optimal = True
            dprint(f"Optimal objective value: {obj_value}\n")
        elif model.Status == 3 or model.Status == 4:
            raise Exception("Model is infeasible")
        else:
            optimal = False
            dprint(f"Best found (possibly not optimal) objective value = {obj_value}\n")

        # model and solution information
        model_information = numb_vars_bin, numb_vars_cont, numb_constr, creation_time
        solution_information = obj_value, best_bound, optimal, solving_time, logfile

        # Store the processed solution data in a dictionary
        PR_assignment = {p: "" for p in instance.patient_ids}
        for (p, r) in valid_y_indices:
            if abs(y[p, r].X - 1) < 10e-6:  # possible rounding error
                PR_assignment[p] = r

        NR_assignment = {(n, s): [] for n in instance.nurse_ids for s in instance.nurses[n]["shifts"]}
        for (n, r, s) in valid_x_indices:
            if abs(x[n, r, s].X - 1) < 10e-6:  # possible rounding error
                NR_assignment[n, s].append(r)
        solution = (PR_assignment, NR_assignment)

        # # print values
        # for v in model.getVars():
        #     print(f"{v.VarName} = {v.X}")

        #### COMPUTE THE OBJECTIVE VALUES
        # continuity = sum(ever_assigned[n, p].X for n in instance.nurse_ids for p in instance.patient_ids)
        # gender = sum(gender_vio[r, s].X for r in instance.room_ids for s in instance.early_shifts)
        # skill = sum(skill_vio[p, s].X for p in instance.patient_ids for s in instance.patients[p]["shifts"])
        # workload = sum(load_vio[n, s].X for n in instance.nurse_ids for s in instance.nurses[n]["shifts"])
        # imbalance = sum(max_load[s].X - min_load[s].X for s in instance.all_shifts)
        # print(f"Continuity: {continuity}")
        # print(f"Gender: {gender}")
        # print(f"Skill: {skill}")
        # print(f"Workload: {workload}")
        # print(f"Imbalance: {imbalance}")


    else:
        model_information = numb_vars_bin, numb_vars_cont, numb_constr, creation_time
        solution_information = None, None, None, None, None
        solution = (None,None)
    return model_information, solution_information, solution
            
class DefaultCallback:
    """Making the Gurobi Logfiles more precise in terms of runtime reporting"""
    def __init__(self, instance_path):
        current_time = datetime.datetime.now()
        day, month = current_time.day, current_time.month
        hour, minute, second, microsecond = current_time.hour, current_time.minute, current_time.second, current_time.microsecond
        logfile_name = instance_path + f"_log_{day}-{month}_{hour}-{minute}-{second}.{microsecond}" + ".txt"

        self.logfile_name = logfile_name
        self.logfile = open(logfile_name, "w")

        self.line_number = 0
        self.table_start_line_number = -1
        self.reached_table_header = False
        self.inside_table = False

        self.reached_no_rel_heur = False

        self._best = None
        self._bound = None


    def __call__(self, model, where):
        if where == GRB.Callback.MIPSOL:
            incumbent = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            bestbd = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
            self._best = incumbent
            self._bound = bestbd

        if where == GRB.Callback.MESSAGE:
            self.line_number += 1
            callback_msg =  model.cbGet(GRB.Callback.MSG_STRING)
            runtime = model.cbGet(GRB.Callback.RUNTIME)

            # Finding the table header
            if not self.reached_table_header:
                if len(callback_msg.split()) > 0:
                    if callback_msg.split()[0] == "Nodes" and callback_msg.split()[-1] == "Work":
                        # we have reached the table header
                        self.reached_table_header = True
                        self.table_start_line_number = self.line_number + 3

            # Determining when inside the table
            if self.line_number == self.table_start_line_number:
                # we have reached the first line of the table
                self.inside_table = True

            # Determining when outside the table again
            if self.inside_table:
                if callback_msg == "\n":
                    self.inside_table = False

            # Determine when stopped with noRelHeur
            if self.reached_no_rel_heur:
                if callback_msg == "NoRel heuristic complete\n":
                    self.reached_no_rel_heur = False

            # Writing to the logfile
            if self.inside_table:
                new_callback_msg = re.sub(r'\d+s$', f"{runtime:.3f}s", callback_msg)
                self.logfile.write(new_callback_msg)
            elif self.reached_no_rel_heur:
                first_word = callback_msg.split()[0]
                second_word = callback_msg.split()[1]
                if first_word not in ["Elapsed", "Transition"]:
                    if second_word != "phase-1":
                        new_callback_msg = f"Found heuristic solution: {self._best:.5f} - {self._bound:.5f} - {runtime:.3f}\n"
                        # new_callback_msg = "Found heuristic solution"
                        # if self._best is not None:
                        #     new_callback_msg += f" {self._best:.5f}"
                        # else:
                        #     new_callback_msg += f" None"
                        # if self._bound is not None:
                        #     new_callback_msg += f" - {self._bound:.5f}"
                        # new_callback_msg += f" - {runtime:.3f}\n"
                        self.logfile.write(new_callback_msg)
            else:
                self.logfile.write(callback_msg)

            # Determine when starting the noRelHeuristic
            if not self.reached_no_rel_heur:
                if callback_msg == "Starting NoRel heuristic\n":
                    self.reached_no_rel_heur = True


if __name__ == "__main__":
    ##########################
    # Initialize instance
    instance_name = "test03"
    dataset_name = instance_name_to_dataset(instance_name)
    file_path = "./Instances" + "/" + dataset_name + "/" + instance_name + ".json"
    instance = Instance(file_path, print_instance_info=True)

    ##########################
    # Run ILP for a single instance
    solve_it = True
    time_limit = 600
    norel_heur_time = 0
    relax_problem = True
    print_ilp_log = True
    store_ilp_log = False
    logfile_folder = "./logfiles/" # must add this folder yourself to store logfiles
    set_debug(True)
    model_information, solution_information, solution = ILP_formulation(instance, solve_it, time_limit, norel_heur_time,
                                                                        relax_problem, print_ilp_log, store_ilp_log,
                                                                        logfile_folder)
    print(f"Logfile = {solution_information[-1]}")

    ##########################
    # Plot the solution
    # from AnalyzingSolution.SolutionVisualization import plot_room_gender_and_occupancy
    # solution_attributes = get_solution_attributes(instance, solution)
    # plot_room_gender_and_occupancy(instance, solution_attributes)

    ##########################
    # Store the solution in a pickle file
    # solution_filename = name + ".pkl"
    # solution_filepath = "./Solutions" # must add this folder yourself to store solutions
    # pickle_store(solution, solution_filename, solution_filepath)

    ##########################
    # Load the solution from a pickle file and plot solution
    # solution_filename = name + ".pkl"
    # solution_filepath = "./Solutions" # must add this folder yourself to retrieve solutions
    # solution = pickle_load(solution_filename, solution_filepath)
    # solution_attributes = get_solution_attributes(instance, solution)
    # plot_room_gender_and_occupancy(instance, solution_attributes)
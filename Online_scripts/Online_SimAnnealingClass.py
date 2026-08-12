from HelperFunctions import copy_solution, instance_name_to_dataset
import numpy as np
import datetime
from Online_SimAnnealingFunctions import *
from Online_ComputeObjective import compute_objective_emergency_LS
import matplotlib.pyplot as plt

class OnlineSimulatedAnnealer:
    def __init__(self, instance):
        self.instance = instance

        ### PARAMETERS
        # Iterations
        self.cur_iter = 0
        self.max_iters = 10
        self.max_iters_per_s_update = dict()

        # temperature parameters
        self.T_start = 2
        self.T_end = 0.06
        self.cur_temperature = self.T_start
        self.cooling_rate = (self.T_end / self.T_start) ** (1 / self.max_iters)

        # move parameters
        self.move_options = ["MovePatient","SwapPatients","ChangeNurse","RemoveNurse"]
        self.parameters_per_s_update = dict()

        # other parameters
        self.do_objective_check = False
        self.store_best_solution = True # set to False to prevent storing the best solution
                                        # (uses cur_solution in next s_update)

        # initial solution
        self.initial_solution_file_name = None
        self.initial_solution_path_name = None

        ### SOLUTION AND OBJECTIVE
        # Solutions
        self.cur_solution = None
        self.best_solution = None

        # Solution attributes
        self.solution_attributes = None

        # Objectives
        self.cur_obj = float('inf')
        self.best_obj = float('inf')

        ### RECORDING PROCESSES
        # Record the process of the objective
        self.cur_obj_process = []
        self.best_obj_process = []
        # self.move_count = {move_type:0 for move_type in self.move_options}

        ### LOOKUP THINGS FOR SPEED
        self.feasible_room_shift_pairs = None
        self.other_nurses_per_shift = None

        ### ONLiNE SPECIFIC THINGS
        self.fixed_nurse_count_per_patient = {p:dict() for p in instance.patient_ids}

        ## RUNTIME
        self.runtime_per_s_update = dict()

    def execute(self):
        """Runs the simulated annealing algorithm for each updating shift"""

        # NOTE: we consider three types of solutions:
        #   - inside the run_SA function, PR_assignment will be of the form {p:r,...} and only consider
        #       the patients which are known and need to be scheduled (solution_local)
        #   - this solution is (outside the run_SA function) transformed into one where also past patients are included
        #       and is stored in solution history (solution_global)
        #   - this solution is then partly taken to be used in the fixed PR and NR assignments. (solution_fixed)

        # reset other optimization things
        self.reset_optimization()

        # determine parameters
        self.parameters_per_s_update = self.determine_params()

        solution_global = None
        fixed_PR_assignment = dict()
        fixed_NR_assignment = dict()
        NR_assignment_local = dict()

        # for easy access
        instance = self.instance

        # store the previous room for occupants in this for the solution attribute creation
        PR_assignment_local = {p: instance.patients[p]["prev_room"] for p in instance.occupant_ids}

        solution_history = dict()
        print(f"\t\tFinished update shifts:", end=" ")
        for i, s_update in enumerate(instance.schedule_update_shifts):
            # update parameters
            self.max_iters = self.max_iters_per_s_update[s_update]
            parameters = self.parameters_per_s_update[s_update]
            self.T_start = parameters[0]
            self.T_end = parameters[1]
            self.move_prob_vector = parameters[2]

            # get intial solution
            self.initialize_solution(s_update, PR_assignment_local, NR_assignment_local)
            self.initialize_objective(s_update, fixed_PR_assignment, fixed_NR_assignment)

            start_time = datetime.datetime.now()
            self.run_SA(s_update)
            end_time = datetime.datetime.now()
            runtime = (end_time - start_time).total_seconds()
            self.runtime_per_s_update[s_update] = runtime

            # print(f"current objective = {self.cur_obj:.3f}")
            # print(f"best objective = {self.best_obj:.3f}")

            if self.store_best_solution:
                solution_local = self.best_solution
                local_obj = self.best_obj
            else:
                # if we do not want to repeatedly copy a solution, use the final solution
                solution_local = self.cur_solution
                local_obj = self.cur_obj
            PR_assignment_local, NR_assignment_local = solution_local

            # define the next update shift
            if i < len(instance.schedule_update_shifts)-1:
                s_next_update = instance.schedule_update_shifts[i + 1]
            else:
                s_next_update = 3 * instance.numberOfDays

            # transform the local solution into a global one and update the fixed values
            temp_PR_assignment = fixed_PR_assignment.copy()
            temp_NR_assignment = fixed_NR_assignment.copy()
            for p in PR_assignment_local:
                r = PR_assignment_local[p]
                for s in instance.patients[p]["shifts"]:
                    if s in instance.early_shifts:
                        if s >= s_update:
                            temp_PR_assignment[p, s] = r
                            if s < s_next_update:
                                fixed_PR_assignment[p, s] = r
            for (n, s) in NR_assignment_local:
                temp_NR_assignment[n, s] = NR_assignment_local[n, s]
                if s < s_next_update:
                    fixed_NR_assignment[n, s] = NR_assignment_local[n, s]

            # update the fixed nurse count
            for s in range(s_update, s_next_update):
                for p in instance.known_patients_per_shift[s, s_update]:
                    r = PR_assignment_local[p] # stay in same room between s_update and s_next_update

                    for n in instance.nurses_per_shift[s]:
                        if r in NR_assignment_local[n, s]:
                            # nurse is assigned to p
                            if n not in self.fixed_nurse_count_per_patient[p]:
                                self.fixed_nurse_count_per_patient[p][n] = 1
                            else:
                                self.fixed_nurse_count_per_patient[p][n] += 1
            # print(f"self.fixed_nurse_count_per_patient = {self.fixed_nurse_count_per_patient}")

            # add the global solution to the solution history
            solution_global = (temp_PR_assignment, temp_NR_assignment)
            solution_history[s_update] = solution_global

            # see if calculated objective matches with true objective
            if self.do_objective_check:
                obj_value, _ = compute_objective_emergency_LS(instance, solution_global, False)

                if abs(obj_value - local_obj) > 10e-6:
                    raise Exception(f"Incorrect final objective value: {obj_value} != {local_obj} = local_obj")
            print(s_update, end=", ")
        print()

        return solution_global, solution_history

    def run_SA(self, s_update):
        """
        Runs the simulated annealing algorithm in update shift s_update
        """

        while self.cur_iter < self.max_iters:
            # if self.cur_iter % 1000 == 0:
            #     print(f"Iteration {self.cur_iter}")

            move_prob_vector = self.move_prob_vector

            move_available = False
            while not move_available:
                # Choose a move type
                move_index = np.random.choice(4, p = move_prob_vector)
                move_type = self.move_options[move_index]
                # print(move_type)

                # Apply the move
                move_available = True
                if move_type == "MovePatient":
                    self.do_MovePatient_move(s_update)
                elif move_type == "SwapPatients":
                    move_available = self.do_SwapPatients_move(s_update)
                elif move_type == "ChangeNurse":
                    self.do_ChangeNurse_move(s_update)
                else:
                    move_available = self.do_RemoveNurse_move(s_update)

                if not move_available:
                    # distribute the probability of the failed move across the other moves (according to ratios)
                    problematic_move_prob = move_prob_vector[move_index]
                    new_move_prob_vector = []
                    for i in range(4):
                        if i == move_index:
                            new_move_prob_vector.append(0)
                        else:
                            new_move_prob_vector.append(move_prob_vector[i] / (1 - problematic_move_prob))
                    move_prob_vector = new_move_prob_vector

            # if self.best_obj < self.best_obj_process[-1]:
            #     print(f"Improved objective: new best = {self.best_obj}, prev current = {self.cur_obj_process[-1]}")

            # update processes
            self.cur_obj_process.append(self.cur_obj)
            self.best_obj_process.append(self.best_obj)

            # update temperature and iteration counter
            self.cur_temperature *= self.cooling_rate
            self.cur_iter += 1

    def do_MovePatient_move(self, s_update):
        """
        Will uniformly select a (non-occupant) patient p,
        then uniformly select a room r (different from current room).

        THIS FUNCTION ASSUMES THAT EVERY PATIENT HAS AT LEAST TWO FEASIBLE ROOMS AND THAT THEY CAN THEREFORE
        BE ASSIGNED TO A DIFFERENT ROOM. IT HAS NO FAILSAFE FOR THIS AND WILL LIKELY CAUSE AN ERROR.
        """

        # retrieve things for fast lookups
        instance = self.instance
        schedulable_patients = instance.schedulable_patients[s_update]
        solution_attributes = self.solution_attributes

        # randomly select a patient
        p_index = np.random.choice(len(schedulable_patients))
        p = schedulable_patients[p_index]

        # determine the rooms they cannot be moved to
        PR_assignment = self.cur_solution[0]
        r1 = PR_assignment[p]
        incomp_rooms = instance.patients[p]["incompatible_rooms"]
        room_candidates = []
        for r in instance.room_ids:
            if r != r1:
                if r not in incomp_rooms:
                    room_candidates.append(r)

        if len(room_candidates) == 0:
            raise Exception("No possible room to change to")

        # select the room to move to
        r2_index = np.random.choice(len(room_candidates))
        r2 = room_candidates[r2_index]
        # print(f"Move patient {p} from room {r1} to room {r2}")

        # determine the change in objective
        obj_delta = delta_eval_MovePatient(instance, s_update, p, r1, r2, solution_attributes)

        # always accept the move if it improves, otherwise accept with a probability
        if obj_delta <= 0:
            accept_move = True
        else:
            acceptance_probability = np.exp(-obj_delta / self.cur_temperature)
            uniform_sample = np.random.uniform(0, 1)
            if uniform_sample < acceptance_probability:
                accept_move = True
            else:
                accept_move = False

        if accept_move:
            # update the solution
            (PR_assignment, NR_assignment) = self.cur_solution
            PR_assignment[p] = r2
            self.cur_solution = (PR_assignment, NR_assignment)

            # update solution attributes
            self.solution_attributes = update_solution_attributes_MovePatient(instance, s_update, p, r1, r2,
                                                                              PR_assignment, solution_attributes)

            # update the objective
            self.cur_obj += obj_delta
            if self.cur_obj < self.best_obj:
                self.best_obj = self.cur_obj
                if self.store_best_solution: # improves speed when we only care about the objective
                    self.best_solution = copy_solution(self.cur_solution)

    def do_SwapPatients_move(self, s_update):
        """
        Will uniformly select a pair (p1,p2) where they overlap in shifts.
        The pair can only swap if their current room is compatible for the other and are in different rooms.
        """
        # retrieve things for fast lookup
        PR_assignment = self.cur_solution[0]
        instance = self.instance
        solution_attributes = self.solution_attributes

        # retrieve possible patient pairs to swap
        swappable_patient_pairs = solution_attributes["swappable_patient_pairs"]

        if len(swappable_patient_pairs) == 0:
            # if no patient pair could be selected
            return False

        # randomly select a pair and determine their current rooms
        pair_index = np.random.choice(len(swappable_patient_pairs))
        (p1,p2) = swappable_patient_pairs[pair_index]
        r1 = PR_assignment[p1]
        r2 = PR_assignment[p2]

        obj_delta = delta_eval_SwapPatients(instance, s_update, p1, p2, r1, r2, solution_attributes)

        if obj_delta <= 0:
            accept_move = True
        else:
            acceptance_probability = np.exp(-obj_delta / self.cur_temperature)
            uniform_sample = np.random.uniform(0, 1)
            if uniform_sample < acceptance_probability:
                accept_move = True
            else:
                accept_move = False

        if accept_move:
            # update the solution
            (PR_assignment, NR_assignment) = self.cur_solution
            PR_assignment[p1] = r2
            PR_assignment[p2] = r1
            self.cur_solution = (PR_assignment, NR_assignment)

            # update solution attributes
            self.solution_attributes = update_solution_attributes_SwapPatients(instance, s_update, p1, p2, r1, r2,
                                                                               PR_assignment, solution_attributes)

            # update the objective
            self.cur_obj += obj_delta
            if self.cur_obj < self.best_obj:
                self.best_obj = self.cur_obj
                if self.store_best_solution:  # improves speed when we only care about the objective
                    self.best_solution = copy_solution(self.cur_solution)

    def do_ChangeNurse_move(self, s_update):
        """
        Will uniformly select a (r,s) pair where the nurse can be changed,
        then uniformly select another nurse that can be assigned to the room
        """
        # create local variables for fast lookup
        instance = self.instance
        solution_attributes = self.solution_attributes
        nurse_per_room = solution_attributes["nurse_per_room"]

        # determine in which rooms and shifts we can change the nurse
        feasible_room_shift_pairs = self.feasible_room_shift_pairs

        # select a random (r,s) pair
        index = np.random.choice(len(feasible_room_shift_pairs))
        (r,s) = feasible_room_shift_pairs[index]
        n1 = nurse_per_room[r,s]

        # determine the nurses it could possibly switch to
        other_nurses = self.other_nurses_per_shift[n1, s]

        # randomly select a random other nurse
        n2_index = np.random.choice(len(other_nurses))
        n2 = other_nurses[n2_index]

        # Calculate the change in objective
        obj_delta = delta_eval_ChangeNurse(instance, s_update, r, s, n1, n2, solution_attributes)

        # always accept the move if it improves, otherwise accept with a probability
        if obj_delta <= 0:
            accept_move = True
        else:
            acceptance_probability = np.exp(-obj_delta / self.cur_temperature)
            uniform_sample = np.random.uniform(0, 1)
            if uniform_sample < acceptance_probability:
                accept_move = True
            else:
                accept_move = False


        if accept_move:
            # update the solution
            (PR_assignment, NR_assignment) = self.cur_solution
            NR_assignment[n1, s].remove(r)
            NR_assignment[n2, s].append(r)
            self.cur_solution = (PR_assignment, NR_assignment)

            # update solution attributes
            self.solution_attributes = update_solution_attributes_ChangeNurse(instance, s_update, r, s, n1, n2,
                                                                              solution_attributes)

            # update the objective
            self.cur_obj += obj_delta
            if self.cur_obj < self.best_obj:
                self.best_obj = self.cur_obj
                if self.store_best_solution: # improves speed when we only care about the objective
                    self.best_solution = copy_solution(self.cur_solution)

    def do_RemoveNurse_move(self, s_update):
        """
        Will uniformly select a (p,n) pair to remove that nurse from consideration for that patient.
        For each shift that that nurse is used, select a random other nurse (in the set of nurses for p)
        """
        feasible_moves = self.get_feasible_RemoveNurse_moves(s_update)

        if len(feasible_moves) == 0:
            return False

        # create local variables for fast lookup
        instance = self.instance
        solution_attributes = self.solution_attributes
        PR_assignment = self.cur_solution[0]

        # select a random (p,n) pair
        move_index = np.random.choice(len(feasible_moves))
        (p, r, n1, candidate_assignments) = feasible_moves[move_index]

        # For each shift, randomly select a nurse (from the nurse set of patient p)
        new_assignments = []
        for (s, nurses_available) in candidate_assignments:
            nurse_index = np.random.choice(len(nurses_available))
            n2 = nurses_available[nurse_index]
            new_assignments.append((s, n2))

        # determine the change in objective
        obj_delta = delta_eval_RemoveNurse(instance, s_update, p, r, n1, new_assignments, PR_assignment, solution_attributes)

        # always accept the move if it improves, otherwise accept with a probability
        if obj_delta <= 0:
            accept_move = True
        else:
            acceptance_probability = np.exp(-obj_delta / self.cur_temperature)
            uniform_sample = np.random.uniform(0, 1)
            if uniform_sample < acceptance_probability:
                accept_move = True
            else:
                accept_move = False

        if accept_move:
            # update the solution
            (PR_assignment, NR_assignment) = self.cur_solution
            for (s, n2) in new_assignments:
                NR_assignment[n1, s].remove(r)
                NR_assignment[n2, s].append(r)
            self.cur_solution = (PR_assignment, NR_assignment)

            # update solution attributes
            self.solution_attributes = update_solution_attributes_RemoveNurse(instance, r, n1, new_assignments, solution_attributes)

            # update current and best objective
            self.cur_obj += obj_delta
            if self.cur_obj < self.best_obj:
                self.best_obj = self.cur_obj
                if self.store_best_solution:  # improves speed when we only care about the objective
                    self.best_solution = copy_solution(self.cur_solution)

        return True

    def get_feasible_RemoveNurse_moves(self, s_update):
        solution_attributes = self.solution_attributes
        instance = self.instance
        fixed_nurse_count_per_patient = solution_attributes["fixed_nurse_count_per_patient"]
        nurse_count_per_patient = solution_attributes["nurse_count_per_patient"]
        PR_assignment = self.cur_solution[0]
        NP_assignment = solution_attributes["NP_assignment"]

        RemoveNurse_options = solution_attributes["RemoveNurse_options"]
        patients_to_update = solution_attributes["patients_to_update"]

        for p in patients_to_update:
            patient_shifts = [s for s in instance.patients[p]["shifts"] if s >= s_update]
            patient_nurse_count = nurse_count_per_patient[p]
            options_for_p = []

            # for each assigned nurse, what happens if we cannot use that nurse anymore
            for n in patient_nurse_count:
                if n in fixed_nurse_count_per_patient[p]:
                    # nurse assignment already fixed
                    continue

                can_be_covered = True
                candidate_assignments = []
                for s in patient_shifts:
                    if NP_assignment[p, s] == n:  # if nurse assigned in shift
                        # is there another nurse that can cover the shift
                        other_nurse_available = []
                        for n0 in patient_nurse_count:
                            if n0 != n:
                                if n0 in instance.nurses_per_shift[s]:
                                    other_nurse_available.append(n0)
                        if len(other_nurse_available) == 0:
                            can_be_covered = False
                            break
                        else:
                            candidate_assignments.append((s, other_nurse_available))

                if can_be_covered and len(candidate_assignments) > 1:  # want to remove single changes
                    options_for_p.append((n,candidate_assignments))
            RemoveNurse_options[p] = options_for_p

        self.solution_attributes["RemoveNurse_options"] = RemoveNurse_options
        self.solution_attributes["patients_to_update"] = list()

        feasible_moves = []
        for p in RemoveNurse_options:
            r = PR_assignment[p]
            for (n,candidate_assignments) in RemoveNurse_options[p]:
                feasible_moves.append((p, r, n, candidate_assignments))
        return feasible_moves

    def initialize_solution(self, s_update, prev_PR_assignment, prev_NR_assignment):
        """
        Creates an initial solution to be used in the simulated annealing algorithm.
        If s_update = 0, the solution is generated fully randomly
        Otherwise, the initial solution uses the best solution from the previous s_update and only randomly
            initializes the room for new emergency patients.
        """
        instance = self.instance
        if s_update == 0:
            # randomly assign all patients
            PR_assignment = dict()
            for p in instance.occupant_ids:
                r = instance.patients[p]["prev_room"]
                PR_assignment[p] = r

            for p in instance.known_patients[s_update]:
                if p not in PR_assignment:
                    comp_rooms = []
                    for r in instance.room_ids:
                        if r not in instance.patients[p]["incompatible_rooms"]:
                            comp_rooms.append(r)
                    r_index = np.random.choice(len(comp_rooms))
                    r = comp_rooms[r_index]
                    PR_assignment[p] = r

            # randomly assign the nurses
            NR_assignment = {(n,s): [] for n in instance.nurse_ids
                             for s in instance.nurses[n]["shifts"]}
            for r in instance.room_ids:
                for s in instance.all_shifts:
                    nurses_present = instance.nurses_per_shift[s]
                    nurse_index = np.random.choice(len(nurses_present))
                    n = nurses_present[nurse_index]
                    NR_assignment[n,s].append(r)
        else:
            # keep previously scheduled patients in the same room, rest are randomly assigned
            new_patients = instance.unexpected_patients_per_update_shift[s_update]
            PR_assignment = dict()
            for p in instance.schedulable_patients[s_update]:
                if p in new_patients:
                    comp_rooms = []
                    for r in instance.room_ids:
                        if r not in instance.patients[p]["incompatible_rooms"]:
                            comp_rooms.append(r)
                    r_index = np.random.choice(len(comp_rooms))
                    r = comp_rooms[r_index]
                    PR_assignment[p] = r
                else:
                    PR_assignment[p] = prev_PR_assignment[p]

            # keep the NR_assignment the same
            NR_assignment = dict()
            for s in instance.all_shifts:
                if s >= s_update:
                    for n in instance.nurses_per_shift[s]:
                        NR_assignment[n,s] = prev_NR_assignment[n,s]


        # get solution and solution attributes
        self.cur_solution = (PR_assignment, NR_assignment)
        if self.store_best_solution:
            self.best_solution = copy_solution(self.cur_solution)
        self.solution_attributes = get_SA_solution_attributes(instance, self.cur_solution, s_update, prev_PR_assignment, self.fixed_nurse_count_per_patient)

        # initialize other parameters
        self.cur_iter = 0
        self.cur_temperature = self.T_start
        self.cooling_rate = (self.T_end / self.T_start) ** (1 / self.max_iters)

        # determine all (r,s) pairs (for the ChangeNurse moves)
        feasible_room_shift_pairs = []
        for r in instance.room_ids:
            for s in instance.all_shifts:
                if s >= s_update:
                    if len(instance.nurses_per_shift[s]) >= 2:
                        feasible_room_shift_pairs.append((r, s))
        self.feasible_room_shift_pairs = feasible_room_shift_pairs

        # determine for each (n,s) pair the list of nurses n' != n that are present during s (for the ChangeNurse moves)
        # this can remain the same for all s_update
        if self.other_nurses_per_shift is None:
            other_nurses_per_shift = {}
            for s in instance.all_shifts:
                for n1 in instance.nurses_per_shift[s]:
                    other_nurses_per_shift[n1, s] = []
                    for n2 in instance.nurses_per_shift[s]:
                        if n1 != n2:
                            other_nurses_per_shift[n1, s].append(n2)
            self.other_nurses_per_shift = other_nurses_per_shift

    def initialize_objective(self, s_update, fixed_PR_assignment, fixed_NR_assignment):
        # turn the local solution from self.cur_solution into a global one (for the objective value computation)
        PR_assignment_global = fixed_PR_assignment.copy()
        NR_assignment_global = fixed_NR_assignment.copy()
        PR_assignment_local, NR_assignment_local = self.cur_solution
        for p in PR_assignment_local:
            r = PR_assignment_local[p]
            for s in self.instance.patients[p]["shifts"]:
                if s in self.instance.early_shifts:
                    if s >= s_update:
                        PR_assignment_global[p, s] = r
        for (n, s) in NR_assignment_local:
            NR_assignment_global[n, s] = NR_assignment_local[n, s]
        solution_global = (PR_assignment_global, NR_assignment_global)


        # compute the objective
        obj_value, _ = compute_objective_emergency_LS(self.instance, solution_global, False)
        # print("^Initial objective\n")

        # store the objective
        self.cur_obj = obj_value
        self.best_obj = obj_value

        # Update processes
        self.cur_obj_process.append(self.cur_obj)
        self.best_obj_process.append(self.best_obj)

    def determine_params(self):
        params_per_s_update = dict()
        instance = self.instance

        for s_update in instance.schedule_update_shifts:
            numb_patients_to_schedule = len(instance.schedulable_patients[s_update])
            if numb_patients_to_schedule < 75:
                T_start = 2
                T_end = 0.06
                move_prob_vector = (0.2, 0.1, 0.6, 0.1)  # must sum to 1
            elif numb_patients_to_schedule < 240:
                T_start = 2
                T_end = 0.2
                move_prob_vector = (0.1, 0.25, 0.45, 0.2)  # must sum to 1
            else:
                T_start = 2
                T_end = 0.09
                move_prob_vector = (0.15, 0.2, 0.45, 0.2)  # must sum to 1
            params = (T_start, T_end, move_prob_vector)
            params_per_s_update[s_update] = params
        return params_per_s_update

    def plot_process(self):
        # Show full process
        plt.plot(self.cur_obj_process, label="Current solution", alpha=0.8)
        plt.plot(self.best_obj_process, label="Best solution", alpha=0.8)
        plt.title(f"Optimization process\nInstance {self.instance.instance_name}")
        plt.show()

        # zoomed in plot
        lowest_y_value = min(self.best_obj_process)
        lowest_final_obj = min(self.best_obj_process[-self.max_iters:])
        plt.axhline(y=lowest_final_obj, color="red", linestyle="--", alpha=0.3)
        plt.plot(self.cur_obj_process, label="Current solution", alpha=0.8)
        plt.plot(self.best_obj_process, label="Current solution", alpha=0.8)
        plt.ylim([0.95*lowest_y_value, 1.55*lowest_final_obj])
        plt.title(f"Optimization process (zoomed in)\nInstance {self.instance.instance_name}")
        plt.show()

    def reset_optimization(self):
        ### PARAMETERS
        # Iterations
        self.cur_iter = 0

        ### SOLUTION AND OBJECTIVE
        # Solutions
        self.cur_solution = None
        self.best_solution = None
        self.solution_attributes = None

        # Objectives
        self.cur_obj = float('inf')
        self.best_obj = float('inf')

        ### RECORDING PROCESSES
        # Record the process of the objective
        self.cur_obj_process = []
        self.best_obj_process = []
        # self.move_count = {move_type:0 for move_type in self.move_options}

        ### LOOKUP THINGS FOR SPEED
        self.feasible_room_shift_pairs = None
        self.other_nurses_per_shift = None
        self.fixed_nurse_count_per_patient = {p: dict() for p in self.instance.patient_ids}

        ## RUNTIME
        self.runtime_per_s_update = dict()

if __name__ == "__main__":
    from Online_InstanceClass import EmergencyInstance
    from Online_ComputeObjective import compute_objective_emergency

    instance_name = "m01_10_1"
    dataset_name = instance_name_to_dataset(instance_name)
    file_path = "Instances/" + dataset_name + "/" + instance_name + ".json"
    instance = EmergencyInstance(file_path, print_instance_info=False)


    np.random.seed(42) # set the seed for replicability
    simulated_annealer = OnlineSimulatedAnnealer(instance)
    simulated_annealer.max_iters_per_s_update = {s_update: 1000 for s_update in instance.schedule_update_shifts}

    print(f"Running SA for instance {instance_name}")
    print(f"\tmax_iters: {simulated_annealer.max_iters_per_s_update}")

    start_time = datetime.datetime.now()
    solution, solution_history = simulated_annealer.execute()
    end_time = datetime.datetime.now()
    print(f"\t\t runtime per s_update {simulated_annealer.runtime_per_s_update}")
    print(f"\t=> In total took {(end_time - start_time).total_seconds()} seconds")
    print(f"\t=> Found objective {simulated_annealer.best_obj}")

    compute_objective_emergency_LS(instance, solution, print_table=True)
    simulated_annealer.plot_process()


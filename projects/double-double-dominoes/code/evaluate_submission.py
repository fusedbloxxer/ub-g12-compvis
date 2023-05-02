def compare_annotations_regular_tasks(filename_predicted,filename_gt,verbose=0):
	p = open(filename_predicted,"rt")  	
	gt = open(filename_gt,"rt")  	
	all_lines_p = p.readlines()
	all_lines_gt = gt.readlines()

	#positions and values	
	number_lines_p = len(all_lines_p)
	number_lines_gt = len(all_lines_gt)

	match_positions = 1
	match_values = 1
	match_score = 1

	for i in range(number_lines_gt-1):		
		current_pos_gt, current_value_gt = all_lines_gt[i].split()
		
		if verbose:
			print(i)
			print(current_pos_gt,current_value_gt)

		try:
			current_pos_p, current_value_p = all_lines_p[i].split()
			
			if verbose:
				print(current_pos_p,current_value_p)

			if(current_pos_p != current_pos_gt):
				match_positions = 0
			if(current_value_p != current_value_gt):
				match_letters = 0	
		except:
			match_positions = 0
			match_letters = 0		
	try:
		#verify if there are more positions + values lines in the prediction file
		current_pos_p, current_value_p = all_lines_p[i+1].split()
		match_positions = 0
		match_values = 0

		if verbose:
			print("EXTRA LINE:")
			print(current_pos_p,current_value_p)
			
	except:
		pass



	points_positions = 0.015 * match_positions
	points_values = 0.015 * match_values	

	#scores
	last_line_p = all_lines_p[-1]
	score_p = last_line_p.split()
	last_line_gt= all_lines_gt[-1]
	score_gt = last_line_gt.split()
	
	if verbose:
		print(score_p,score_gt)

	if(score_p != score_gt):
		match_score = 0

	points_score = 0.015 * match_score

	return points_positions, points_values,points_score


def compare_annotations_bonus_task(filename_predicted,filename_gt,verbose=0):
	p = open(filename_predicted,"rt")  	
	gt = open(filename_gt,"rt")
	all_lines_p = p.readlines()
	all_lines_gt = gt.readlines()
	
	number_lines_p = len(all_lines_p)
	number_lines_gt = len(all_lines_gt)

	match = 1

	for i in range(number_lines_gt-1):		
		current_string_gt = all_lines_gt[i].split()
		current_string_p = all_lines_p[i].split()

		if verbose:
			print(i)
			print(current_string_gt,current_string_p)

		if(current_string_gt != current_string_p):
			match = 0
				


	points_bonus = 0.05 * match
	

	return points_bonus

#change this on your machine pointing to your results (txt files)
predictions_path_root = "/home/invokariman/Projects/git/ub-g12-compvis/projects/double-double-dominoes/data/train/output/"

#change this on your machine to point to the ground-truth test
gt_path_root = "/home/invokariman/Projects/git/ub-g12-compvis/projects/double-double-dominoes/data/train/truth/"



#change this to 1 if you want to print results at each move
verbose = 1
total_points_regular_tasks = 0

#regular tasks
for game in range(1,6):
	for move in range(1,21):
		
		name_move = str(move)
		if(move< 10):
			name_move = '0'+str(move)

		filename_predicted = predictions_path_root + "regular_tasks//" + str(game) + '_' + name_move + '.txt'
		filename_gt = gt_path_root + "regular_tasks//" + str(game) + '_' + name_move + '.txt'

		game_move = str(game) + '_' + name_move
		points_position = 0
		points_values = 0
		points_score = 0		

		try:
			points_position, points_values, points_score = compare_annotations_regular_tasks(filename_predicted,filename_gt,verbose)
		except Exception as e:
			print("For image: ", game_move, " encountered an error")

		print("Image: ", game_move, "Points position: ", points_position, "Points values: ",points_values, "Points score: ", points_score)
		total_points_regular_tasks = total_points_regular_tasks + points_position + points_values + points_score
print("Regular tasks: ",total_points_regular_tasks)


total_points_bonus_task = 0
#bonus task
for img in range(1,11):
	print(img)
	name_img = str(img)
	if(img < 10):
		name_img = '0'+str(img)

	print("name_img = ",name_img)
	filename_predicted = predictions_path_root + "bonus_task//" + name_img + '.txt'
	filename_gt = gt_path_root + "bonus_task//" + name_img + '.txt'

				
	points_bonus = 0
	try:
		points_bonus = compare_annotations_bonus_task(filename_predicted,filename_gt,verbose)
	except:
		print("For image: ", name_img, " encountered an error")

	print("Image: ", name_img, "Points bonus: ", points_bonus)
	total_points_bonus_task = total_points_bonus_task + points_bonus

print("Bonus task: ",total_points_bonus_task)


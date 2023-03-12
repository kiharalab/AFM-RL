def calculate_clashes():
	pass

def build_interactions(no_chains):
    interaction_list = []
    for i in range(0,no_chains):
        for j in range(i,no_chains):
            if i==j:continue
            interaction_list.append([i,j])
    return interaction_list
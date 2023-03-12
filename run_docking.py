if __name__=="__main__":
	parser = argparse.ArgumentParser(description='MCTS docking')
	parser.add_argument('--num_sims', action="store", required=True, type=int)
	args=parser.parse_args()
	root_node=Node(State())

	new_node=UCTSEARCH(args.num_sims,root_node)
	print("Num Children: %d"%len(new_node.children))
	for i,c in enumerate(new_node.children):
		print(i,c)
	print("Best Child: %s"%new_node.state)
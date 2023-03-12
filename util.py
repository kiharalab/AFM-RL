class State():
	NUM_ACTIONS = 1000	
	GOAL = 0 # current best protein score
	MOVES=[]#[2,-2,3,-3] A-B, A-C comformation changes/ interactions
	MAX_VALUE = -inf #(5.0*(NUM_TURNS-1)*NUM_TURNS)/2 # not used
	num_moves=len(MOVES) # remaining interactions, should be set in init
	ALL_INTERACTION = [] # all possible interaction for states
	def __init__(self, value=0, moves=[], turn=NUM_TURNS):
		self.value=value
		self.turn=turn # set of actions maybe? random generator from 1 to 1000
		self.moves=moves # the rest of comformation possible from this state
	def next_state(self):
		nextmove=random.choice([x for x in self.MOVES]) # randomly select a child
		nextstate=State(0, self.moves-ALL_INTERACTION,1000) # 0 reward, moves -
		return nextstate
	def terminal(self):
		if self.moves == 0: # should be changed to moves
			return True
		return False
	def reward(self):
		protein_score = calculate_protein_like_score(self.path)
		if protein_score <= GOAL:
			r = 1
		else:
			r = -1
		return r
	def __hash__(self):
		return int(hashlib.md5(str(self.moves).encode('utf-8')).hexdigest(),16)
	def __eq__(self,other):
		if hash(self)==hash(other):
			return True
		return False
	def __repr__(self):
		s="Value: %d; Moves: %s"%(self.value,self.moves)
		return s

class Node():
	def __init__(self, state, parent=None):
		self.visits=1
		self.reward=0.0	
		self.state=state
		self.children=[]
		self.parent=parent	
	def add_child(self,child_state):
		child=Node(child_state,self)
		self.children.append(child)
	def update(self,reward):
		self.reward+=reward
		self.visits+=1
	def fully_expanded(self):
		if len(self.children)==self.state.num_moves:
			return True
		return False
	def __repr__(self):
		s="Node; children: %d; visits: %d; reward: %f"%(len(self.children),self.visits,self.reward)
		return s

def Q_table_init(states,actions):
	# initialize Q(s,a)
	Q = {}
	for s in states:
		Q[s] = {}
  		for a in actions: # actions is fixed, 1k pools of decoy
    	Q[s][a] = 0

	# initial Q values for all states in grid
	print(Q)
	return Q

def UCTSEARCH(time_steps,root):
	for iter in range(int(time_steps)):
		front=TREEPOLICY(root)
		reward=DEFAULTPOLICY(front.state)
		BACKUP(front,reward)
		
		if iter in [10,20,30,40,50,100,200,300,500]:
			logger.info("simulation: %d"%iter)
			logger.info(root)
	
	return BESTCHILD(root,0)

def TREEPOLICY(node):
	#a hack to force 'exploitation' in a game where there are many options, and you may never/not want to fully expand first
	while node.state.terminal()==False:
		if len(node.children)==0:
			return EXPAND(node)
		elif random.uniform(0,1)<.5:
			node=BESTCHILD(node,SCALAR)
		else:
			if node.fully_expanded()==False:	
				return EXPAND(node)
			else:
				node=BESTCHILD(node,SCALAR)
	return node

def EXPAND(node):
	tried_children=[c.state for c in node.children]
	new_state=node.state.next_state()
	while new_state in tried_children:
		new_state=node.state.next_state()
	node.add_child(new_state)
	return node.children[-1]

#current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
def BESTCHILD(node,scalar):
	bestscore=0.0
	bestchildren=[]
	for c in node.children:
		exploit=c.reward/c.visits
		explore=math.sqrt(2.0*math.log(node.visits)/float(c.visits))	
		score=exploit+scalar*explore
		if score==bestscore:
			bestchildren.append(c)
		if score>bestscore:
			bestchildren=[c]
			bestscore=score
	if len(bestchildren)==0:
		logger.warn("OOPS: no best child found, probably fatal")
	return random.choice(bestchildren)

def DEFAULTPOLICY(state):
	while state.terminal()==False:
		state=state.next_state()
	return state.reward()

def BACKUP(node,reward):
	while node!=None:
		node.visits+=1
		node.reward+=reward
		node=node.parent
	return


def Docking_Order(node):
	order_list = []
	while node!=None:
		order_list.append(node)
		node=node.parent
	print order_list
	return order_list
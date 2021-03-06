import org.vu.contest.ContestSubmission;
import org.vu.contest.ContestEvaluation;

import java.util.Random;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Properties;

public class player100 implements ContestSubmission
{
	Random rnd_;
	ContestEvaluation evaluation_;
    private int evaluations_limit_;
    static int populationSize = 50;
    static int parentsSize = 20; //needs to be even number and >= than 2*populationSize when (lambda, mu) selection is used
	static double s = 1.5;  //parameter for rank based selection
	static double mutationVariation = 0.1; //parameter for mutation variation
	static double sigma_mating = 5.0; //speciation distance (min;max) is (0;32) 
	static double crossOver = 0.5; //cross-over weight between (0;1)
	static int dimension = 10;
	double[] parent_fitness;
	
	public player100()
	{
		rnd_ = new Random();
	}
	
	public static void main(String[] args) {
		
	}
	
	public void setSeed(long seed)
	{
		// Set seed of algortihms random process
		rnd_.setSeed(seed);
	}

	public void setEvaluation(ContestEvaluation evaluation)
	{
		// Set evaluation problem used in the run
		evaluation_ = evaluation;
		
		// Get evaluation properties
		Properties props = evaluation.getProperties();
        // Get evaluation limit
        evaluations_limit_ = Integer.parseInt(props.getProperty("Evaluations"));
		// Property keys depend on specific evaluation
		// E.g. double param = Double.parseDouble(props.getProperty("property_name"));
        boolean isMultimodal = Boolean.parseBoolean(props.getProperty("Multimodal"));
        boolean hasStructure = Boolean.parseBoolean(props.getProperty("Regular"));
        boolean isSeparable = Boolean.parseBoolean(props.getProperty("Separable"));
		// Do sth with property values, e.g. specify relevant settings of your algorithm
        if(isMultimodal && hasStructure) {
 
        }
        if(isMultimodal && !hasStructure){

        }else{
            // Do sth else
        }
    }
	
	
    /*
     * CREATE INITAL POPULATION
     */
	public double[][] init_population(int populationSize) {
		//Create 2-dimensional array, which is populationSize X 10 and continuous input in [-5,5]
		//Do this randomly with 10 dimensional uniform distribution
		Random generate = new Random();
		double[][] pop = new double[populationSize][dimension];

		for(int x = 0; x < populationSize; x++) {
			for(int y = 0; y<dimension; y++) {
				pop[x][y] = (generate.nextDouble() * 10) - 5;
			}
		}
		return(pop);
	}
	
	
	/*
	 * FITNESS SHARING
	 */
	public double [] fitness_sharing(double [][] arr, double [] fitn) {
		double[][] difference = new double[populationSize][dimension];
		double [] distance = new double [populationSize];
		double[] sharedfitness = new double[populationSize];
		int a = 1;
		double sigmashare = 0.5/Math.pow(10.0,1.0/10.0);
		for (int i = 0; i < populationSize; i++){
			double mi = 0;
			for (int j = 0; j < populationSize; j++){
				double sum = 0;
				for (int k = 0; k < dimension; k++){
					//take the normalized square of the difference
					difference[j][k] = (((arr[i][k] - arr[j][k])/10) * ((arr[i][k] - arr[j][k])/10));
				}
				for (int l = 0; l < dimension; l++){
					sum += difference[j][l];
				}
				if (Math.sqrt(sum) <= sigmashare) {
					distance[j] = (1-Math.pow((Math.sqrt(sum)/sigmashare),a));
				}
				else{
					distance[j] = 0;
				}
			}
			for (int m = 0; m < populationSize; m++){
				mi += distance[m];
			}
			sharedfitness[i] = (fitn[i] / mi);
		}
		return(sharedfitness);
	}
	
	/*
	 * SELECTION OPERATORS
	 */
	//Select parents from population with linear rank roulette wheel style
	public double[][] rank_roulette_wheel_speciation(double[][] population, double[] pop_fitness, int parents) {
		orderByFitness(pop_fitness, population);
		int mu = pop_fitness.length;
		
		double[][] parents_pop = new double[parents][10];
		double totalWeight  = 0.0d;
		double[] fitness_store = new double[parents];
		double[] rank_value = new double[mu];
		parent_fitness = fitness_store;
		
		//make rank probabilities
		int rank_mu = mu-1;
		for(int i = 0; i<mu; i++) {
			rank_value[i] = (2-s)/rank_mu + 2*i*(s-1)/(rank_mu*(rank_mu-1));
		}
		
		
		//Calculate total fitness of population
		for(int i = 0; i < mu; i++) {
			totalWeight += rank_value[i];
		}
		
		//Select random individual with respect to fitness value
		for(int j = 0; j < parents; j++) {
			int randomIndex = -1;
			double random = Math.random() * totalWeight;
			for(int i = 0; i < rank_value.length; i++) {
				random -= rank_value[i];
				if(random<=0.0d) {
					randomIndex = i;
					break;
				}
			}
			parent_fitness[j] = pop_fitness[randomIndex];
			parents_pop[j] = population[randomIndex];
			if(j%2==0) {
				int index = mating(randomIndex, population, mu);
				if(index>=0) {
					parent_fitness[j+1] = pop_fitness[index];
					parents_pop[j+1] = population[index];
					j++;
				}
			}
			
		}
		return parents_pop;
	}	
	
	public double[][] rank_roulette_wheel(double[][] population, double[] pop_fitness, int parents) {
			orderByFitness(pop_fitness, population);
			int mu = pop_fitness.length;
			
			double[][] parents_pop = new double[parents][10];
			double totalWeight  = 0.0d;
			double[] fitness_store = new double[parents];
			double[] rank_value = new double[mu];
			parent_fitness = fitness_store;
			
			//make rank probabilities
			int rank_mu = mu-1;
			for(int i = 0; i<mu; i++) {
				rank_value[i] = (2-s)/rank_mu + 2*i*(s-1)/(rank_mu*(rank_mu-1));
			}
			
			
			//Calculate total fitness of population
			for(int i = 0; i < mu; i++) {
				totalWeight += rank_value[i];
			}
			
			//Select random individual with respect to fitness value
			for(int j = 0; j < parents; j++) {
				int randomIndex = -1;
				double random = Math.random() * totalWeight;
				for(int i = 0; i < rank_value.length; i++) {
					random -= rank_value[i];
					if(random<=0.0d) {
						randomIndex = i;
						break;
					}
				}
				parent_fitness[j] = pop_fitness[randomIndex];
				parents_pop[j] = population[randomIndex];
			}
			return parents_pop;
		}
	
	//Select parents from population with roulette wheel style
	public double[][] roulette_wheel(double[][] population, double[] pop_fitness, int parents) {
		double[][] parents_pop = new double[parents][10];
		double totalWeight  = 0.0d;
		double[] fitness_store = new double[parents];
		parent_fitness = fitness_store;
		
		//Calculate total fitness of population
		for(int i = 0; i < pop_fitness.length; i++) {
			totalWeight += pop_fitness[i];
		}
		
		//Select random individual with respect to fitness value
		for(int j = 0; j < parents; j++) {
			int randomIndex = -1;
			double random = Math.random() * totalWeight;
			for(int i = 0; i < pop_fitness.length; i++) {
				random -= pop_fitness[i];
				if(random<=0.0d) {
					randomIndex = i;
					break;
				}
			}
			parent_fitness[j] = pop_fitness[randomIndex];
			parents_pop[j] = population[randomIndex];
		}
		return parents_pop;
	}
	
	
	/*
	 * RECOMBINATION OPERATORS
	 */
	//Do cross-over by taking average with random weight alpha
	public double[][] crossOver(double[][] population, int number_childs) {
		double[][] childs = new double[number_childs][10];
		
		for(int i = 0; i<parentsSize; i+=2) {
			//double alpha = Math.random();
			for(int j = 0; j<10; j++) {
				childs[i/2][j] = population[i][j]*crossOver + population[i+1][j]*(1-crossOver);
			}
		}
		return childs;
	}
	
	
	/*
	 * MUTATION OPERATORS
	 */
	public void gaussianMutation(double[][] pop) {
		Random gaussian = new Random();
		for(int i = 0; i<parentsSize/2; i++) {
			for(int j = 0; j<10; j++) {
				pop[i][j] = pop[i][j] + gaussian.nextGaussian() * mutationVariation;
			}
		}
	}
	
	
	/*
	 * REPLACEMENT OPERATORS
	 */
	//replace population with other population (mu + lambda) selection
		public void fitness_replacement_plus(double[] fitness_parents, double[] fitness_childs, double[][] pop_parents, double[][] pop_childs) {
			double[] new_fitness = new double[fitness_parents.length+fitness_childs.length];
			double[][] new_pop = new double[fitness_parents.length+fitness_childs.length][10];
			
			for(int i=0; i<fitness_parents.length; i++) {
				new_fitness[i] = fitness_parents[i];
				new_pop[i] = pop_parents[i];
			}
			for(int i=0; i<fitness_childs.length; i++) {
				new_fitness[fitness_parents.length+i] = fitness_childs[i];
				new_pop[fitness_parents.length+i] = pop_childs[i];
			}
			
			orderByFitness(new_fitness, new_pop);
			for(int i=0; i<populationSize;i++) {
				fitness_parents[i] = new_fitness[fitness_parents.length+fitness_childs.length-1-i];
				pop_parents[i] = new_pop[fitness_parents.length+fitness_childs.length-1-i];
			}
			
			//System.out.println(fitness_parents.length);
			//System.out.println(fitness_childs.length);
			
			//orderByFitness(fitness_parents, pop_parents);
			//System.out.println("ala");
            //orderByFitness(fitness_childs, pop_childs);
			//
			//for(int i = fitness_parents.length-1; i > 0; i--) {
			//	for(int j = 0; j<fitness_parents.length; j++) {
			//		int validation = Double.compare(fitness_parents[j], fitness_childs[i]);
			//		if(validation < 0) {
			//			fitness_parents[j] = fitness_childs[i];
			//			pop_parents[j] = pop_childs[i];
			//			break;
			//		}
			//	}
			//}
		}
		
		//replace population with other population (mu, lambda) selection
		public void fitness_replacement(double[] pop_fitness, double[] fitness_childs,  double[][] population, double[][] pop_childs) {
			orderByFitness(fitness_childs, pop_childs);
			
			for(int i=0; i<populationSize; i++) {
				population[i] = pop_childs[fitness_childs.length-1-i];
				pop_fitness[i] = fitness_childs[fitness_childs.length-1-i];
			}
			
		}
	
	
	/*
	 * GENERAL METHODS
	 */
	//Calculate fitness
	public double[] calculate_fitness(double[][] population, int population_number) {
		double pop_fitness[] = new double[population_number];
        for(int x = 0; x<population_number; x++) {
        	pop_fitness[x] = (double) evaluation_.evaluate(population[x]);
        }
        return pop_fitness;
	}
	
	//order population and fitness from low - high fitness
	public void orderByFitness(double[] fitness, double[][] population) {
		//rank fitness from low to high
		Integer[] rank = new Integer[fitness.length];
    	for(int i = 0; i < fitness.length; i++) rank[i] = i;
    	Arrays.sort(rank, new Comparator<Integer>(){
    		@Override
    	    public int compare(Integer o1, Integer o2){
    	    	return Double.compare(fitness[o1], fitness[o2]);
    	    }
    	});
    	
    	//order fitness and population w.r.t. index from rank
    	double temp_fitness[] = new double[rank.length];
    	double temp_pop[][] = new double[rank.length][10];
    	
    	for(int i = 0; i<rank.length; i++) {
    		temp_fitness[rank[i]] = fitness[i];
    		temp_pop[rank[i]] = population[i];
    	}
    	
    	for(int i = 0; i<rank.length; i++) {
    		fitness[i] = temp_fitness[i];
    		population[i] = temp_pop[i];
    	}
	}
	
	//mating for speciation
	public int mating(int randomIndex, double[][] population, int mu) {
		ArrayList<Integer> matings = new ArrayList<Integer>();
		for(int i=0; i<mu; i++) {
			if(i != randomIndex) {
				double sum = 0.0;
				for(int j=0; j<10; j++) {
					sum = sum+Math.pow((population[i][j] - population[randomIndex][j]), 2.0);
				}
				sum = Math.sqrt(sum);
				if(sum <= sigma_mating) {
					matings.add(i);
				}
			}
		}
		if(matings.isEmpty()) {
			return(-1);
		}else {
			int index = (int) (Math.random() * matings.size());
			return(index);
		}
	}
	
	public void run()
	{	
		populationSize = (int) Double.parseDouble(System.getProperty("var1"));
	    parentsSize = (int) Double.parseDouble(System.getProperty("var2")); 
		s = Double.parseDouble(System.getProperty("var3")); 
		mutationVariation = Double.parseDouble(System.getProperty("var4"));
		sigma_mating = Double.parseDouble(System.getProperty("var5")); 
		crossOver = Double.parseDouble(System.getProperty("var6"));
		
		
		// Run your algorithm here
        int evals = 0;
        
        // init population
        double[][] population = init_population(populationSize);
        
        // calculate fitness of population
        double pop_fitness[];
        pop_fitness = calculate_fitness(population, populationSize);

        while(evals<evaluations_limit_){
            // Select parents
        	double[] fitness_sharing = fitness_sharing(population, pop_fitness);
        	double[][] parents = rank_roulette_wheel_speciation(population, fitness_sharing, parentsSize);
        	
            // Apply crossover
        	double[][] children;
        	children = crossOver(parents, parentsSize/2);
        	
        	//Apply mutation
        	gaussianMutation(children);
        	
        	//Calculate fitness children for fitness based selection
        	double[] children_fitness;
        	children_fitness = calculate_fitness(children, parentsSize/2);
                    
            // population selection (lambda,mu)
            fitness_replacement_plus(pop_fitness, children_fitness, population, children);
            evals++;;
        }

	}
}

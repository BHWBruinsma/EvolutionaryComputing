import org.vu.contest.ContestSubmission;
import org.vu.contest.ContestEvaluation;

import java.util.Random;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Properties;

public class player100 implements ContestSubmission
{
	Random rnd_;
	ContestEvaluation evaluation_;
    private int evaluations_limit_;
    static int populationSize = 30;
    static int parentsSize = 60;
	static int number_childs = 30;
	double[] parent_fitness;
	double beta;
	double gamma;
	boolean go=false;
	
	public player100()
	{
		rnd_ = new Random();
	}
	
	public static void main(String[] args) {
		System.out.println("Start");
		
		player100 awesome = new player100();
		double[][] pop = awesome.init_population(populationSize, 10);
		double[] fitness = {10.1,9.,8.,7.,10.,5.,4.,3.,2.,1.};
		
		for (double[] x : pop)
		{
		   for (double y : x)
		   {
		        System.out.print(y + " ");
		   }
		   System.out.println();
		}
		System.out.println("middle");
		pop = awesome.select_parents(pop, fitness, parentsSize);
		//System.out.println(pop[0].length);
		//print population for testing
		//awesome.orderByFitness(fitness, pop);
		for (double[] x : pop)
		{
		   for (double y : x)
		   {
		        System.out.print(y + " ");
		   }
		   System.out.println();
		}
		//System.out.println(pop[0].length);
		System.out.println("next");
		
		double[][] children;
    	children = awesome.crossOver(pop, number_childs);
		
    	for (double[] x : children)
		{
		   for (double y : x)
		   {
		        System.out.print(y + " ");
		   }
		   System.out.println();
		}
    	double[] x = {1.,2.,10.,3.,2.5,4.,9.,6.,7.,1.};
    	double[] y = {7.,8.,1.,2.,3.,4.,5.,9.,1.1,2.1};
    	
    	Integer[] idxs = new Integer[x.length];
    	for(int i = 0; i < x.length; i++) idxs[i] = i;
    	Arrays.sort(idxs, new Comparator<Integer>(){
    		@Override
    	    public int compare(Integer o1, Integer o2){
    	    	return Double.compare(y[o1], y[o2]);
    	    }
    	});
    	
    	for(int i = 0; i<10; i++) {
    		System.out.println(y[idxs[i]]);
    		System.out.println(x[idxs[i]]);
    	}
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
        System.out.println(isMultimodal);
        System.out.println(hasStructure);
        System.out.println(isSeparable);
		// Do sth with property values, e.g. specify relevant settings of your algorithm
        if(isMultimodal && hasStructure) {
        	beta = 0.35;
        	gamma = 0.035;
        	populationSize = 80;
            parentsSize = 160;
        	number_childs = 80;
        	go=true;
        }
        if(isMultimodal && !hasStructure){
        	beta = 0.9;
        	gamma = 0.09;
        	populationSize = 50;
            parentsSize = 100;
        	number_childs = 50;
        	go=true;
        }else{
            // Do sth else
        }
    }
    
	//Create initial population
	public double[][] init_population(int populationSize, int dimension) {
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
	
	//Select parents from population with roulette wheel style
	public double[][] select_parents(double[][] population, double[] pop_fitness, int parents) {
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
	
	//Do cross-over by taking average with random weight alpha
	public double[][] crossOver(double[][] population, int number_childs) {
		double[][] childs = new double[number_childs][10];
		double[] estimated_fitness = new double[number_childs];
		
		for(int i = 0; i<parentsSize; i+=2) {
			double alpha = Math.random();
			estimated_fitness[i/2] = parent_fitness[i]*alpha + parent_fitness[i+1]*(1-alpha);
			for(int j = 0; j<10; j++) {
				childs[i/2][j] = population[i][j]*alpha + population[i+1][j]*(1-alpha);
			}
		}
		parent_fitness = estimated_fitness;
		return childs;
	}
	
	public void gaussianMutationVariation(double[][] pop) {
		Random gaussian = new Random();
		for(int i = 0; i<number_childs; i++) {
			for(int j = 0; j<10; j++) {
				int value = (int)parent_fitness[i];
				pop[i][j] = pop[i][j] + gaussian.nextGaussian() * (beta - gamma * value);
			}
		}
	}
	
	public void gaussianMutation(double[][] pop) {
		Random gaussian = new Random();
		for(int i = 0; i<number_childs; i++) {
			for(int j = 0; j<10; j++) {
				pop[i][j] = pop[i][j] + gaussian.nextGaussian() * 0.1;
			}
		}
	}
	
	//Calculate fitness
	public double[] calculate_fitness(double[][] population, int population_number) {
		double pop_fitness[] = new double[population_number];
        for(int x = 0; x<population_number; x++) {
        	pop_fitness[x] = (double) evaluation_.evaluate(population[x]);
        }
        return pop_fitness;
	}
	
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
	
	public void replace(double[] fitness_parents, double[] fitness_childs, double[][] pop_parents, double[][] pop_childs) {
		for(int i = fitness_parents.length-1; i > 0; i--) {
			for(int j = 0; j<fitness_parents.length; j++) {
				int validation = Double.compare(fitness_parents[j], fitness_childs[i]);
				if(validation < 0) {
					fitness_parents[j] = fitness_childs[i];
					pop_parents[j] = pop_childs[i];
					break;
				}
			}
		}
	}
	
	public void run()
	{
		System.out.println(evaluations_limit_);
		// Run your algorithm here
        
        int evals = 0;
        // init population
        double[][] population = init_population(populationSize, 10);
        // calculate fitness
        double pop_fitness[];
        pop_fitness = calculate_fitness(population, populationSize);

        //double pop_fitness[] = new double[populationSize];
        //for(int x = 0; x<populationSize; x++) {
        //	pop_fitness[x] = (double) evaluation_.evaluate(population[x]);
        //}
        System.out.println("start");
        int i = 0;
        while(evals<evaluations_limit_){
        	System.out.println(i);
            // Select parents
        	population = select_parents(population, pop_fitness, parentsSize);
            // Apply crossover / mutation operators
        	double[][] children;
        	children = crossOver(population, number_childs);
        	if(go == true) {
        		gaussianMutationVariation(children);
        	}else {
        		gaussianMutation(children);
        	}
            //double child[] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
            //Check fitness of unknown fuction
            //Double fitness = (double) evaluation_.evaluate(child);
        	double[] children_fitness;
        	children_fitness = calculate_fitness(children, number_childs);
            evals++;;
            // Select survivors (Replace on basis of fitness)
            orderByFitness(pop_fitness, population);
            orderByFitness(children_fitness, children);
            replace(pop_fitness, children_fitness, population, children);
            //population = children;
            //pop_fitness = children_fitness;
            i++;
        }

	}
}

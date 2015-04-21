
package jsat.classifiers.bayesian.graphicalmodel;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;

import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.classifiers.bayesian.ConditionalProbabilityTable;
import jsat.classifiers.bayesian.NaiveBayes;
import jsat.exceptions.FailedToFitException;
import jsat.utils.IntSet;
import static java.lang.Math.*;

/**
 * A class for representing a Baysian Network (BN) for discrete variables. A BN use a graph to representing 
 * the relations between variables, and these links are called the structure. The structure of a BN must be
 * specified by an expert using the {@link #depends(int, int) } method. The target class should be specified 
 * as the parent of the variables which have a causal relationship to it. These children of the target class
 * should then have their own children specified. Once the structure has been specified, the network can be 
 * trained and used for classification. <br>
 * If the network structure has not been specified, or has no relationships for the target class, the BN will
 * create an edge from the target class to every variable. If no edges were ever specified, this initialization
 * of edges corresponds to a {@link NaiveBayes} implementation. 
 * 
 * @author Edward Raff
 */
public class DiscreteBayesNetwork implements Classifier
{

	private static final long serialVersionUID = 2980734594356260141L;
	/**
     * The directed Graph that represents this BN
     */
    protected DirectedGraph<Integer> dag;
    /**
     * The Conditional probability tables for each variable
     */
    protected Map<Integer, ConditionalProbabilityTable> cpts;
    /**
     * The class we are predicting
     */
    protected CategoricalData predicting;
    /**
     * The prior probabilities of each class value 
     */
    protected double[] priors;
    private boolean usePriors = DEFAULT_USE_PRIORS;
    
    /**
     * Whether or not the classifier should take into account the prior probabilities. Default value is {@value #DEFAULT_USE_PRIORS}. 
     */
    public static final boolean DEFAULT_USE_PRIORS = true;

    public DiscreteBayesNetwork()
    {
        dag = new DirectedGraph<Integer>();
    }
    
    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults cr = new CategoricalResults(predicting.getNumOfCategories());
        
        int classId = data.numCategoricalValues();
        //Use log proababilities to avoid underflow
        double logPSum = 0;
        double[] logProbs = new double[cr.size()];
        for(int i = 0; i < cr.size(); i++)
        {
            DataPointPair<Integer> dpp = new DataPointPair<Integer>(data, i);
            for(int classParent : dag.getChildren(classId))
                logProbs[i] += log(cpts.get(classParent).query(classParent, dpp));
            
            if(usePriors)
                logProbs[i] += log(priors[i]);
            logPSum += logProbs[i];
        }
        
        for(int i = 0; i < cr.size(); i++)
            cr.setProb(i, exp(logProbs[i]-logPSum));
        
        return cr;
    }
    
    /**
     * Adds a dependency relation ship between two variables that will be in the network. The integer value corresponds 
     * the the index of the i'th  categorical variable, where the class target's value is the number of categorical variables. 
     * 
     * @param parent the parent variable, which will be explained in part by the child
     * @param child the child variable, which contributes to the conditional probability of the parent. 
     */
    public void depends(int parent, int child)
    {
        dag.addNode(child);
        dag.addNode(parent);
        dag.addEdge(parent, child);
    }

    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        trainC(dataSet);
    }

    public void trainC(ClassificationDataSet dataSet)
    {
        int classID = dataSet.getNumCategoricalVars();
        if(classID == 0 )
            throw new FailedToFitException("Network needs categorical attribtues to work");
        
        predicting = dataSet.getPredicting();
        priors = dataSet.getPriors();
        cpts = new HashMap<Integer, ConditionalProbabilityTable>();
        Set<Integer> cptTrainSet = new IntSet();
        
        if(dag.getNodes().isEmpty())
        {
            for(int i = 0; i < classID; i++)
                depends(classID, i);
        }
        
        for(int classParent : dag.getChildren(classID))
        {
            Set<Integer> depends = dag.getChildren(classParent);
            ConditionalProbabilityTable cpt = new ConditionalProbabilityTable();
            
            cptTrainSet.clear();
            cptTrainSet.addAll(depends);
            cptTrainSet.add(classParent);
            cptTrainSet.add(classID);
            cpt.trainC(dataSet, cptTrainSet);
            cpts.put(classParent, cpt);
        }
    }

    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public Classifier clone()
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }
    
}

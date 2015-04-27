package jsat.regression;

import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.DataPoint;

/**
 * Creates a regressor that averages the results of several voting regression methods. 
 * Null values are not supported, and will cause errors at a later time. The averaged 
 * regressor can be trained, and will train each of its voting regressors. If each 
 * regressor is of the same type, training may not be advisable. 
 * 
 * @author Edward Raff
 */
public class AveragedRegressor implements Regressor
{

	private static final long serialVersionUID = 8870461208829349608L;
	/**
     * The array of voting regressors 
     */
    protected Regressor[] voters;

    /**
     * Constructs a new averaged regressor using the given array of voters
     * @param voters the array of voters to use
     */
    public AveragedRegressor(Regressor... voters)
    {
        if(voters == null ||voters.length == 0)
            throw new RuntimeException("No voters given for construction");
        this.voters = voters;
    }
    
    /**
     * Constructs a new averaged regressor using the given list of voters. 
     * The list of voters will be copied into a new space, so the list may
     * safely be reused. 
     * @param voters the array of voters to use 
     */
    public AveragedRegressor(List<Regressor> voters)
    {
        if(voters == null || voters.isEmpty())
            throw new RuntimeException("No voters given for construction");
        this.voters = voters.toArray(new Regressor[0]);
    }
    
    public double regress(DataPoint data)
    {
        double r = 0.0;
        for(Regressor vote : voters)
            r += vote.regress(data);
        return r / voters.length;
    }

    public void train(RegressionDataSet dataSet, ExecutorService threadPool)
    {
        for(Regressor voter : voters)
            voter.train(dataSet, threadPool);
    }

    public void train(RegressionDataSet dataSet)
    {
        for(Regressor voter :  voters)
            voter.train(dataSet);
    }

    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public AveragedRegressor clone()
    {
        Regressor[] clone = new Regressor[this.voters.length];
        for(int i = 0; i < clone.length; i++)
            clone[i] = voters[i].clone();
        return new AveragedRegressor(clone);
    }
    
}

package jsat.datatransform;

import java.util.Arrays;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

/**
 * A transform for applying a polynomial transformation on the data set. As the 
 * dimension of the data set grows, the number of new features created by a 
 * polynomial transform grows rapidly. It is recommended only for small 
 * dimension problems using small degrees. 
 * 
 * @author Edward Raff
 */
public class PolynomialTransform implements DataTransform
{

    private static final long serialVersionUID = -5332216444253168283L;
    private int degree;

    /**
     * Creates a new polynomial transform of the given degree
     * @param degree the degree of the polynomial
     * @throws ArithmeticException if the degree is not greater than 1
     */
    public PolynomialTransform(int degree)
    {
        if(degree < 2)
            throw new ArithmeticException("The degree of the polynomial was a nonsense value: " + degree);
        this.degree = degree;
    }

    @Override
    public void fit(DataSet data)
    {
        //no-op, nothing needs to be done
    }
    
    @Override
    public DataPoint transform(DataPoint dp)
    {
        Vec x = dp.getNumericalValues();
        int[] setTo = new int[x.length()];
        
        //TODO compute final size directly isntead of doing a pre loop
        int finalSize = 0;
        
        int curCount = increment(setTo, degree, 0);
        do
        {
            finalSize++;
            curCount = increment(setTo, degree, curCount);
        }
        while(setTo[x.length()-1] <= degree);
        
        
        
        
        double[] newVec = new double[finalSize];
        Arrays.fill(newVec, 1.0);
        int index = 0;
        
        Arrays.fill(setTo, 0);
        curCount = increment(setTo, degree, 0);
        do
        {
            for(int i = 0; i < setTo.length; i++)
                if(setTo[i] > 0)
                    newVec[index] *= Math.pow(x.get(i), setTo[i]);
            index++;
            curCount = increment(setTo, degree, curCount);
        }
        while(setTo[x.length()-1] <= degree);
        
        return new DataPoint(new DenseVector(newVec), dp.getCategoricalValues(),
                dp.getCategoricalData(), dp.getWeight());
    }
    
    /**
     * Increments the array to contain representation of the next combination of
     * values in the polynomial
     * 
     * @param setTo the array of values marking how many multiples of that value
     * will be used in construction of the point
     * @param max the degree of the polynomial  
     * @param curCount the current sum of all counts in the array <tt>setTo</tt>
     * @return the new value of <tt>curCount</tt>
     */
    private int increment(int[] setTo, int max, int curCount)
    {
        setTo[0]++;
        curCount++;
        
        if(curCount <= max)
            return curCount;
        
        int carryPos = 0;
        
        while(carryPos < setTo.length-1 && curCount > max)
        {
            curCount-=setTo[carryPos];
            setTo[carryPos] = 0;
            setTo[++carryPos]++;
            curCount++;
        }
        
        return curCount;
    }

    @Override
    public DataTransform clone()
    {
        return new PolynomialTransform(degree);
    }
    
    /**
     * Sets the degree of the polynomial to transform the input vector into
     *
     * @param degree the positive degree to use
     */
    public void setDegree(int degree)
    {
        if (degree < 1)
            throw new IllegalArgumentException("Degree must be a positive value, not " + degree);
        this.degree = degree;
    }

    /**
     * Returns the polynomial degree to use
     *
     * @return the polynomial degree to use
     */
    public int getDegree()
    {
        return degree;
    }
}

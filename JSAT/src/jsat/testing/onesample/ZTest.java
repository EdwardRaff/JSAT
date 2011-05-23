
package jsat.testing.onesample;

import jsat.distributions.Normal;
import jsat.linear.Vec;
import jsat.text.GreekLetters;

/**
 *
 * @author Edward Raff
 */
public class ZTest implements OneSampleTest
{
    private Normal norm;
    
    private double sampleMean;
    private double sampleDev;
    private int sampleSize;
    private H1 h1;
    
    /**
     * The mean of the null hypothesis
     */
    private double hypoMean;

    public ZTest()
    {
        this(0, 1, 1);
    }

    
    
    public ZTest(double sampleMean, double sampleDev, int sampleSize)
    {
        this(H1.NOT_EQUAL, sampleMean, sampleDev, sampleSize);
    }
    
    public ZTest(H1 h1, double sampleMean, double sampleDev, int sampleSize)
    {
        this.h1 = h1;
        this.hypoMean = 0;
        this.sampleMean = sampleMean;
        this.sampleDev = sampleDev;
        this.sampleSize = sampleSize;
        this.norm = new Normal();
    }

    public ZTest(Vec data)
    {
        this(data.mean(), data.standardDeviation(), data.length());
    }
    
    public ZTest(H1 h1, Vec data)
    {
        this(h1, data.mean(), data.standardDeviation(), data.length());
    }

    public H1[] validAlternate()
    {
        return new H1[]
                {
                    H1.LESS_THAN, H1.NOT_EQUAL, H1.GREATER_THAN
                };
    }

    public String testName()
    {
        return "One Sample Z-Test";
    }

    public void setTestUsingData(Vec data)
    {
        this.sampleMean = data.mean();
        this.sampleDev = data.standardDeviation();
        this.sampleSize = data.length();
    }

    public String[] getTestVars()
    {
        return new String[]{GreekLetters.bar("x"), GreekLetters.sigma, "n"};
    }

    public void setTestVars(double[] testVars)
    {
        this.sampleMean = testVars[0];
        this.sampleDev = testVars[1];
        this.sampleSize = (int) testVars[2];
    }

    public String getAltVar()
    {
        return GreekLetters.mu + "0";
    }

    public void setAltVar(double altVar)
    {
        this.hypoMean = altVar;
    }

    public double pValue()
    {
        double se = sampleDev/Math.sqrt(sampleSize);
        
        double zScore = (sampleMean-hypoMean)/se;
        
        if(h1 == H1.NOT_EQUAL)
            return norm.cdf(-Math.abs(zScore))*2;
        else if(h1 == H1.LESS_THAN)
            return norm.cdf(zScore);
        else
            return 1-norm.cdf(zScore);
    }

    public void setAltHypothesis(H1 h1)
    {
        this.h1 = h1;
    }

    public String getNullVar()
    {
        return GreekLetters.mu;
    }
    
    
    
}

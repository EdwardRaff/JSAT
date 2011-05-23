
package jsat.testing.onesample;

import jsat.distributions.StudentT;
import jsat.linear.Vec;
import jsat.text.GreekLetters;

/**
 *
 * @author Edward Raff
 */
public class TTest implements OneSampleTest
{
    
    private StudentT tDist;
    private H1 h1;
    
    private double hypothMean;
    
    private double sampleMean;
    private double sampleDev;
    private double sampleSize;

    public TTest(H1 h1, double hypothMean, double sampleMean, double sampleDev, double sampleSize)
    {
        this.h1 = h1;
        this.hypothMean = hypothMean;
        this.sampleMean = sampleMean;
        this.sampleDev = sampleDev;
        this.sampleSize = sampleSize;
        tDist = new StudentT(sampleSize-1);
    }
    
    public TTest(double hypothMean, double sampleMean, double sampleDev, double sampleSize)
    {
        this(H1.NOT_EQUAL, hypothMean, sampleMean, sampleDev, sampleSize);
    }

    public TTest(H1 h1, double hypothMean, Vec data)
    {
        this(h1, hypothMean, data.mean(), data.standardDeviation(), data.length());
                
    }

    public TTest()
    {
        this(1, 2, 2, 2);
    }
    
    
    public void setTestUsingData(Vec data)
    {
        this.sampleMean = data.mean();
        this.sampleDev = data.standardDeviation();
        this.sampleSize = data.length();
        tDist.setDf(sampleSize-1);
    }

    public String[] getTestVars()
    {
        return new String[]
                {
            GreekLetters.bar("x"), 
            GreekLetters.sigma, 
            "n"
                };
    }

    public void setTestVars(double[] testVars)
    {
        this.sampleMean = testVars[0];
        this.sampleDev = testVars[1];
        this.sampleSize = testVars[2];
        tDist.setDf(sampleSize-1);
    }

    public String getAltVar()
    {
        return GreekLetters.mu + "0";
    }

    public void setAltVar(double altVar)
    {
        hypothMean = altVar;
    }

    public String getNullVar()
    {
        return GreekLetters.mu;
    }

    public H1[] validAlternate()
    {
        return new H1[]
                {
                    H1.LESS_THAN, H1.NOT_EQUAL, H1.GREATER_THAN
                };
    }

    public void setAltHypothesis(H1 h1)
    {
        this.h1 = h1;
    }

    public String testName()
    {
        return "T Test";
    }

    public double pValue()
    {

        double tScore = (sampleMean - hypothMean)*Math.sqrt(sampleSize)/sampleDev;
        
        if(h1 == H1.NOT_EQUAL)
            return tDist.cdf(-Math.abs(tScore))*2;
        else if(h1 == H1.LESS_THAN)
            return tDist.cdf(tScore);
        else
            return 1-tDist.cdf(tScore);
    }
    
}

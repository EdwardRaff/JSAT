
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
    
    private final StudentT tDist;
    private H1 h1;
    
    private double hypothMean;
    
    private double sampleMean;
    private double sampleDev;
    private double sampleSize;

    public TTest(final H1 h1, final double hypothMean, final double sampleMean, final double sampleDev, final double sampleSize)
    {
        this.h1 = h1;
        this.hypothMean = hypothMean;
        this.sampleMean = sampleMean;
        this.sampleDev = sampleDev;
        this.sampleSize = sampleSize;
        tDist = new StudentT(sampleSize-1);
    }
    
    public TTest(final double hypothMean, final double sampleMean, final double sampleDev, final double sampleSize)
    {
        this(H1.NOT_EQUAL, hypothMean, sampleMean, sampleDev, sampleSize);
    }

    public TTest(final H1 h1, final double hypothMean, final Vec data)
    {
        this(h1, hypothMean, data.mean(), data.standardDeviation(), data.length());
                
    }

    public TTest()
    {
        this(1, 2, 2, 2);
    }
    
    
    @Override
    public void setTestUsingData(final Vec data)
    {
        this.sampleMean = data.mean();
        this.sampleDev = data.standardDeviation();
        this.sampleSize = data.length();
        tDist.setDf(sampleSize-1);
    }

    @Override
    public String[] getTestVars()
    {
        return new String[]
                {
            GreekLetters.bar("x"), 
            GreekLetters.sigma, 
            "n"
                };
    }

    @Override
    public void setTestVars(final double[] testVars)
    {
        this.sampleMean = testVars[0];
        this.sampleDev = testVars[1];
        this.sampleSize = testVars[2];
        tDist.setDf(sampleSize-1);
    }

    @Override
    public String getAltVar()
    {
        return GreekLetters.mu + "0";
    }

    @Override
    public void setAltVar(final double altVar)
    {
        hypothMean = altVar;
    }

    @Override
    public String getNullVar()
    {
        return GreekLetters.mu;
    }

    @Override
    public H1[] validAlternate()
    {
        return new H1[]
                {
                    H1.LESS_THAN, H1.NOT_EQUAL, H1.GREATER_THAN
                };
    }

    @Override
    public void setAltHypothesis(final H1 h1)
    {
        this.h1 = h1;
    }

    @Override
    public String testName()
    {
        return "T Test";
    }

    @Override
    public double pValue()
    {

        final double tScore = (sampleMean - hypothMean)*Math.sqrt(sampleSize)/sampleDev;
        
        if(h1 == H1.NOT_EQUAL) {
          return tDist.cdf(-Math.abs(tScore))*2;
        } else if(h1 == H1.LESS_THAN) {
          return tDist.cdf(tScore);
        } else {
          return 1-tDist.cdf(tScore);
        }
    }
    
}

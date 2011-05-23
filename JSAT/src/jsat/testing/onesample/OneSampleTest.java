
package jsat.testing.onesample;

import jsat.linear.Vec;
import jsat.testing.StatisticTest;

/**
 *
 * @author Edward Raff
 */
public interface OneSampleTest extends StatisticTest
{
    /**
     * Sets the statistics that will be tested against an alternate hypothesis. 
     * 
     * @param data 
     */
    public void setTestUsingData(Vec data);
    
    public String[] getTestVars();
    public void setTestVars(double[] testVars);
    
    public String getAltVar();
    public void setAltVar(double altVar);
    
    public String getNullVar();
    
}

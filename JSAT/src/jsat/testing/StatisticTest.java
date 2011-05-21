
package jsat.testing;

/**
 *
 * @author Edward Raff
 */
public interface StatisticTest
{
    public enum H1
    {

        LESS_THAN
        {

            @Override
            public String toString()
            {
                return "<";
            }
        }, 
        GREATER_THAN
        {

            @Override
            public String toString()
            {
                return ">";
            }
        }, 
        NOT_EQUAL
        {

            @Override
            public String toString()
            {
                return "\u2260";
            }
        }
    };

    /**
     * 
     * @return an array of the valid alternate hypothesis for this test
     */
    public H1[] validAlternate();
    
    public void setAltHypothesis(H1 h1);
            
    
    /**
     * 
     * @return a descriptive name for the statistical test
     */
    public String testName();
    
    public double pValue();
    
}

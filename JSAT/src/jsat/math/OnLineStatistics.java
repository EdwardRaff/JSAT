
package jsat.math;

/**
 *
 * This class provides a means of updating summary statistics as each 
 * new data point is added. The data points are not stored. 
 * 
 * @author Edward Raff
 */
public class OnLineStatistics
{
   /**
     * The current mean
     */
   private double mean;
   /**
    * The current number of samples seen
    */
   private int n;
   
   //Intermediat value updated at each step, variance computed from it
   private double m2, m3, m4;

    public OnLineStatistics()
    {
        this(0, 0, 0, 0, 0);
    }

   
   
    public OnLineStatistics(int n, double mean, double variance, double skew, double kurt)
    {
        this.n = n;
        this.mean = mean;
        if(n != 0)
        {
            this.m2 = variance*(n-1);
            this.m3 = Math.pow(m2, 3.0/2.0)*skew/Math.sqrt(n); 
            this.m4 = (3+kurt)*m2*m2/n;
        }
        else
            m2 = m3 = m4 = 0;
    }
   
   
   
   
   public void add(double x)
   {
       //See http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
       
       double n1 = n;
       n++;
       double delta = x - mean;
       double delta_n = delta/n;
       double delta_n2 = delta_n*delta_n;
       double term1 = delta*delta_n*n1;
       
       
       mean += delta_n;
       m4 += term1 * delta_n2 * (n*n - 3*n + 3) + 6 * delta_n2 * m2 - 4 * delta_n * m3;
       m3 += term1 * delta_n * (n - 2) - 3 * delta_n * m2;
       m2 += delta*(x-mean);
   }

   public double getMean()
   {
        return mean;
    }
   
   public double getVarance()
   {
       return m2/(n-1);
   }
   
   public double getStandardDeviation()
   {
       return Math.sqrt(getVarance());
   }
   
   public double getSkewness()
   {
       return Math.sqrt(n) * m3 / Math.pow(m2, 3.0/2.0);
   }
   
   public double getKurtosis()
   {
       return (n*m4) / (m2*m2) - 3;
   }
}

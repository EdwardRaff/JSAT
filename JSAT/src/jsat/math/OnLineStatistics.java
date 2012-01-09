
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

   private Double min, max;
   
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
        min = max  = null;
    }
   
    private OnLineStatistics(int n, double mean, double m2, double m3, double m4, double min, double max)
    {
        this.n = n;
        this.mean = mean;
        this.m2 = m2;
        this.m3 = m3;
        this.m4 = m4;
        this.min = min;
        this.max = max;
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
       
       if(min == null)
           min = max = x;
       else
       {
           min = Math.min(min, x);
           max = Math.max(max, x);
       }
   }
   
   public static OnLineStatistics add(OnLineStatistics A, OnLineStatistics B)
   {
       if(A.n == B.n && B.n == 0)
           return new OnLineStatistics();
       else if(B.n == 0)
           return new OnLineStatistics(A.n, A.mean, A.m2, A.m3, A.m4, A.min, A.max);
       else if(A.n == 0)
           return new OnLineStatistics(B.n, B.mean, B.m2, B.m3, B.m4, B.min, B.max);
       
       int nX = B.n + A.n;
       int nXsqrd = nX*nX;
       int nAnB = B.n*A.n;
       int AnSqrd = A.n*A.n;
       int BnSqrd = B.n*B.n;
       
       double delta = B.mean - A.mean;
       double deltaSqrd = delta*delta;
       double deltaCbd = deltaSqrd*delta;
       double deltaQad = deltaSqrd*deltaSqrd;
       double newMean = (A.n* A.mean + B.n * B.mean)/(A.n + B.n);
       double newM2 = A.m2 + B.m2 + deltaSqrd / nX *nAnB;
       double newM3 = A.m3 + B.m3 + deltaCbd* nAnB*(A.n - B.n) / nXsqrd + 3 * delta * (A.n * B.m2 - B.n * A.m2)/nX;
       double newM4 = A.m4 + B.m4 
               + deltaQad * (nAnB*(AnSqrd - nAnB + BnSqrd)/(nXsqrd*nX)) 
               + 6 * deltaSqrd*(AnSqrd*B.m2 + BnSqrd*A.m2)/nXsqrd
               + 4 * delta *(A.n*B.m3 - B.n*A.m3)/nX;
       
        return new OnLineStatistics(nX, newMean, newM2, newM3, newM4, Math.min(A.min, B.min), Math.max(A.max, B.max));   
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
   
   public double getMin()
   {
       return min;
   }
   
   public double getMax()
   {
       return max;
   }
   
}

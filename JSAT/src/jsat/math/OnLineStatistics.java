
package jsat.math;

/**
 *
 * This class provides a means of updating summary statistics as each 
 * new data point is added. The data points are not stored, and values
 * are updated with an online algorithm. 
 * <br>
 * As such, this class has constant memory usage, regardless of how many
 * values are added. But the results may not be as numerically accurate,
 * and can degrade badly given specific data sequences. 
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
   private double n;
   
   //Intermediat value updated at each step, variance computed from it
   private double m2, m3, m4;

   private Double min, max;
   
   /**
    * Creates a new set of statistical counts with no information
    */
    public OnLineStatistics()
    {
        this(0, 0, 0, 0, 0);
    }

    /**
     * Creates a new set of statistical counts with these initial values, and can then be updated in an online fashion 
     * @param n the total weight of all data points added. This value must be non negative
     * @param mean the starting mean. If <tt>n</tt> is zero, this value will be ignored. 
     * @param variance the starting variance. If <tt>n</tt> is zero, this value will be ignored. 
     * @param skew the starting skewness. If <tt>n</tt> is zero, this value will be ignored. 
     * @param kurt the starting kurtosis. If <tt>n</tt> is zero, this value will be ignored. 
     * @throws ArithmeticException if <tt>n</tt> is a negative number
     */
    public OnLineStatistics(double n, double mean, double variance, double skew, double kurt)
    {
        if(n < 0)
            throw new ArithmeticException("Can not have a negative set of weights");
        this.n = n;
        if(n != 0)
        {
            this.mean = mean;
            this.m2 = variance*(n-1);
            this.m3 = Math.pow(m2, 3.0/2.0)*skew/Math.sqrt(n); 
            this.m4 = (3+kurt)*m2*m2/n;
        }
        else
            this.mean = m2 = m3 = m4 = 0;
        min = max  = null;
    }
   
    private OnLineStatistics(double n, double mean, double m2, double m3, double m4, double min, double max)
    {
        this.n = n;
        this.mean = mean;
        this.m2 = m2;
        this.m3 = m3;
        this.m4 = m4;
        this.min = min;
        this.max = max;
    }
    
    /**
     * Copy Constructor
     * @param other the version to make a copy of
     */
    public OnLineStatistics(OnLineStatistics other)
    {
        this(other.n, other.mean, other.m2, other.m3, other.m4, 
                other.min, other.max);
    }
    
    /**
     * Adds a data sample with unit weight to the counts. 
     * @param x the data value to add
     */
   public void add(double x)
   {
       add(x, 1.0);
   }
   
   /**
    * Adds a data sample the the counts with the provided weight of influence. 
    * @param x the data value to add
    * @param weight the weight to give the value
    * @throws ArithmeticException if a negative weight is given 
    */
   public void add(double x, double weight)
   {
       //See http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
       
       if(weight < 0)
           throw new ArithmeticException("Can not add a negative weight");
       else if(weight == 0)
           return;
       
       double n1 = n;
       n+=weight;
       double delta = x - mean;
       double delta_n = delta*weight/n;
       double delta_n2 = delta_n*delta_n;
       double term1 = delta*delta_n*n1;
       
       
       mean += delta_n;
       m4 += term1 * delta_n2 * (n*n - 3*n + 3) + 6 * delta_n2 * m2 - 4 * delta_n * m3;
       m3 += term1 * delta_n * (n - 2) - 3 * delta_n * m2;
       m2 += weight*delta*(x-mean);
       
       if(min == null)
           min = max = x;
       else
       {
           min = Math.min(min, x);
           max = Math.max(max, x);
       }
   }
   
   /**
    * Effectively removes a sample with the given value and weight from the total. 
    * Removing values that have not been added may yield results that have no meaning
    * <br><br>
    * NOTE: {@link #getSkewness() } and {@link #getKurtosis() } are not currently updated correctly
    * 
    * @param x the value of the sample
    * @param weight the weight of the sample
    * @throws ArithmeticException if a negative weight is given
    */
   public void remove(double x, double weight)
   {
       if(weight < 0)
           throw new ArithmeticException("Can not remove a negative weight");
       else if(weight == 0)
           return;
       
       double n1 = n;
       n-=weight;
       double delta = x - mean;
       double delta_n = delta*weight/n;
       double delta_n2 = delta_n*delta_n;
       double term1 = delta*delta_n*n1;
       
       
       mean -= delta_n;
       
       m2 -= weight*delta*(x-mean);
       //TODO m3 and m4 arent getting updated correctly 
       m3 -= term1 * delta_n * (n - 2+weight) - 3 * delta_n * m2;
       m4 -= term1 * delta_n2 * (n*n - 3*n + 3) + 6 * delta_n2 * m2 - 4 * delta_n * m3;
   }
   
   /**
    * Computes a new set of counts that is the sum of the counts from the given distributions. 
    * <br><br>
    * NOTE: Adding two statistics is not as numerically stable. If A and B have values of similar
    * size and scale, the values of the 3rd and 4th moments {@link #getSkewness() } and 
    * {@link #getKurtosis() } will suffer from catastrophic cancellations, and may not 
    * be as accurate. 
    * @param A the first set of statistics
    * @param B the second set of statistics 
    * @return a new set of statistics that is the addition of the two. 
    */
   public static OnLineStatistics add(OnLineStatistics A, OnLineStatistics B)
   {
       if(A.n == B.n && B.n == 0)
           return new OnLineStatistics();
       else if(B.n == 0)
           return new OnLineStatistics(A.n, A.mean, A.m2, A.m3, A.m4, A.min, A.max);
       else if(A.n == 0)
           return new OnLineStatistics(B.n, B.mean, B.m2, B.m3, B.m4, B.min, B.max);
       
       double nX = B.n + A.n;
       double nXsqrd = nX*nX;
       double nAnB = B.n*A.n;
       double AnSqrd = A.n*A.n;
       double BnSqrd = B.n*B.n;
       
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
   
   /**
    * Returns the sum of the weights for all data points added to the statistics. 
    * If all weights were 1, then this value is the number of data points added. 
    * @return 
    */
   public double getSumOfWeights()
   {
       return n;
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

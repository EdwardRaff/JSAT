
package jsat;

import jsat.distributions.NormalDistribution;
import jsat.math.SpecialMath;

/**
 *
 * @author Edward Raff
 */
public class Main {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args)
    {
        // TODO code application logic here
//        System.out.println(NormalDistribution.cdf(1.27, 0, 1));
//        System.out.println(NormalDistribution.invcdf(0.53, 0, 1));
        System.out.println(SpecialMath.lnGamma(0.23486));
        System.out.println(SpecialMath.lnGamma(163.2164));
    }

}

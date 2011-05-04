
package jsat;

import jsat.distributions.NormalDistribution;
import jsat.math.ContinuedFraction;
import jsat.math.Function;
import jsat.math.SpecialMath;
import jsat.math.integration.Romberg;
import jsat.math.integration.Trapezoidal;

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

        Function func = new Function() {

            public double f(double... x)
            {
                return Math.exp(-x[0]*x[0])*2/Math.sqrt(Math.PI);
            }
        };
//        System.out.println(Trapezoidal.trapz(func, 0, 1, 1000));
//        System.out.println(Romberg.romb(func, 0, 1, 20));
//        System.out.println("0.6321189695429691");
//        System.out.println(0.8427007929497149);
        
        
        ContinuedFraction cf2 = new ContinuedFraction() {

            @Override
            public double getA(int pos, double... args)
            {

                if(pos % 2 == 0)
                {
                    pos /= 2;//the # of the even term
                    
                    return pos*args[1];
                }
                else
                {
                    pos/=2;
                    
                    return -(args[0]+pos)*args[1];
                }
            }

            @Override
            public double getB(int pos, double... args)
            {
                
                return args[0];
            }
        };
        
                

        double a = -11;

        for(double x = 0.5; x < 10.5; x+=0.5)
        {
//            System.out.print(SpecialMath.lnLowIncGamma(a, x) + ",");
            System.out.print(Double.toString(SpecialMath.gammaIncUp(a,x)).replaceAll("E", "*10^") + ",");
//            System.out.print(x + ",");
            
        }
        System.out.println();
        for(double x = 0.5; x < 50; x+=0.5)
        {
//            System.out.print(Double.toString(1-SpecialMath.gammaQ(a, x)).replaceAll("E", "*10^") + ",");
//            System.out.print(Math.exp(SpecialMath.lnLowIncGamma1(a, x)) + ",");
//            double tmp  = SpecialMath.lnLowIncGamma1(a, x);
//            if( Double.isNaN(tmp))
//                System.out.println("x: " + x);
//            SpecialMath.lnLowIncGamma1(a, x);
        }

//        System.out.println(SpecialMath.lnLowIncGamma1(a, 41.5));
        
        System.out.println();
        
//        System.out.println(Math.exp(SpecialMath.lnLowIncGamma(a, a)));
//        System.out.println(3397585.145212605);
//        System.out.println(Math.pow(a, a)*Math.exp(-a)/cf.lentz(a, a));
//        System.out.println(Math.pow(a, a)*Math.exp(-a)/cf.backwardNaive(800, a, a));
        
        
        
    }

}

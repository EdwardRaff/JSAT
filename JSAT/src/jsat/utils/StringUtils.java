package jsat.utils;

/**
 *
 * @author Edward Raff
 */
public class StringUtils
{
    public static int parseInt(CharSequence s, int start, int end, int radix)
    {
        boolean negative = false;
        int val = 0;
        
        for(int i = start; i < end; i++)
        {
            char c = s.charAt(i);
            if (c == '-')
                if (i == start)
                    negative = true;
                else
                    throw new NumberFormatException("Negative sign did not occur at begining of sequence");
            else if (c == '+')
                if (i == start)
                    negative = false;//do nothing really
                else
                    throw new NumberFormatException("Positive sign did not occur at begining of sequence");
            else
            {
                int digit = Character.digit(c, radix);
                if(digit < 0)
                    throw new NumberFormatException("Non digit character '" + c + "' encountered");
                val *= radix;
                val += digit;
            }
        }
        if(negative)
            return -val;
        else
            return val;
    }

    public static int parseInt(CharSequence s, int start, int end)
    {
        return parseInt(s, start, end, 10);
    }
    
    //Float/Double to String follows algo laid out here http://krashan.ppa.pl/articles/stringtofloat/
    
    private enum States
    {
        /**
         * Either '+' or '-', or not sign present
         */
        SIGN,
        /**
         * skipping leading zeros in the integer part of mantissa
         */
        LEADING_ZEROS_MANTISSA,
        /**
         * reading leading zeros in the fractional part of mantissa.
         */
        LEADING_ZEROS_FRAC,
        /**
         * reading integer part of mantissa
         */
        MANTISSA_INT_PART,
        
        /**
         * reading fractional part of mantissa.
         */
        MANTISSA_FRAC_PART,
        
        /**
         * reading sign of exponent
         */
        EXPO_SIGN,
        
        EXPO_LEADING_ZERO,
        
        /**
         * reading exponent digits
         */
        EXPO,
    }
    
    public static double parseDouble(CharSequence s, int start, int end)
    {
        //hack check for NaN at the start
        if((end-start) == 3 && s.length() >= end && s.charAt(start) == 'N')
            if(s.subSequence(start, end).toString().equals("NaN"))
                return Double.NaN;
        States state = States.SIGN;
        int pos = start;
        
        int sign = 1;
        long mantissa = 0;
        /**
         * Mantissa can only be incremented 18 times, then any more will 
         * overflow (2^63-1 ~= 9.2* 10^18
         */
        byte mantisaIncrements = 0;
        int implicitExponent = 0;
        //used for (val)e(val) case
        int expoSign = 1;
        int explicitExponent = 0;
        
        while(pos < end)//run the state machine
        {
            char c = s.charAt(pos);
            switch(state)
            {
                case SIGN:
                    if (c == '-')
                    {
                        sign = -1;
                        pos++;
                    }
                    else if(c == '+')
                        pos++;
                    else if (!Character.isDigit(c))//not a '-', '+', or digit, so error
                        throw new NumberFormatException();
                    state = States.LEADING_ZEROS_MANTISSA;
                    continue;
                case LEADING_ZEROS_MANTISSA:
                    if(c == '0')
                        pos++;
                    else if(c == '.')
                    {
                        pos++;
                        state = States.LEADING_ZEROS_FRAC;
                    }
                    else if (Character.isDigit(c))
                        state = States.MANTISSA_INT_PART;
                    else if(c == 'e' || c == 'E')//could be something like +0e0
                        state = States.MANTISSA_FRAC_PART;//this is where that case is handeled
                    else
                        throw new NumberFormatException();
                    continue;
                case LEADING_ZEROS_FRAC:
                    if(c == '0')
                    {
                        pos++;
                        implicitExponent--;
                    }
                    else if(Character.isDigit(c))
                        state = States.MANTISSA_FRAC_PART;
                    else
                        throw new NumberFormatException();
                    continue;
                case MANTISSA_INT_PART:
                    if (c == '.')
                    {
                        pos++;
                        state = States.MANTISSA_FRAC_PART;
                    }
                    else if (Character.isDigit(c))
                    {
                        if(mantisaIncrements < 18)
                        {
                            mantissa = mantissa * 10 + Character.digit(c, 10);
                            mantisaIncrements++;
                        }
                        else//we are going to lose these, compencate with an implicit *= 10
                            implicitExponent++;
                        pos++;
                    }
                    else
                        state = States.MANTISSA_FRAC_PART;
                    //if we hit a invalid char it will get erred on in FRAC_PART
                    continue;
                case MANTISSA_FRAC_PART:
                    if (Character.isDigit(c))
                    {
                        if (mantisaIncrements < 18)
                        {
                            mantissa = mantissa * 10 + Character.digit(c, 10);
                            implicitExponent--;
                            mantisaIncrements++;
                        }
                        else//we are going to lose these
                        {
                            //we would have incresed the implicit exponent
                            //but we would have subtracted if we could 
                            //so do nothing
                        }
                        pos++;
                    }
                    else if (c == 'e' || c == 'E')
                    {
                        pos++;
                        state = States.EXPO_SIGN;
                    }
                    else
                        throw new NumberFormatException();
                    continue;
                case EXPO_SIGN:
                    if (c == '-')
                    {
                        expoSign = -1;
                        pos++;
                    }
                    else if(c == '+')
                        pos++;
                    else if (!Character.isDigit(c))//not a '-', '+', or digit, so error
                        throw new NumberFormatException();
                    state = States.EXPO_LEADING_ZERO;
                    continue;
                case EXPO_LEADING_ZERO:
                    if(c == '0')
                    {
                        pos++;
                    }
                    else if(Character.isDigit(c))
                        state = States.EXPO;
                    else
                        throw new NumberFormatException();
                    continue;
                case EXPO:
                    if(Character.isDigit(c))
                    {
                        explicitExponent = explicitExponent * 10 + Character.digit(c, 10);
                        pos++;
                    }
                    else 
                        throw new NumberFormatException();
                    continue;
            }
        }
        
        int finalExpo = expoSign*explicitExponent + implicitExponent;
        if(mantissa == 0)//easiest case!
            if (sign == -1)
                return -0.0;
            else
                return 0.0;
        if(finalExpo == 0)//easy case! 
            return sign*mantissa;
        
        return sign * (mantissa*Math.pow(10, finalExpo));
    }
}

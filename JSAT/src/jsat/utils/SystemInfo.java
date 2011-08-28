
package jsat.utils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * This class provides Static system information that may be useful for 
 * algorithms that want to adjust their behavior based on the system's 
 * hardware information. All information supposes that hardware is 
 * consistent, ie: all memory installed is of the same type and all
 * CPUs installed are the same model and stepping. 
 * 
 * @author Edward Raff
 */
public class SystemInfo
{
    
    public final static String OS_String = System.getProperty("os.name");
    
    /**
     * 
     * @return true if this machine is running some version of Windows
     */
    public static boolean isWindows()
    {
        return OS_String.contains("Win");
    }
    
    /**
     * 
     * @return true is this machine is running some version of Mac OS X
     */
    public static boolean isMac()
    {
        return OS_String.contains("Mac");
    }
    
    /**
     * 
     * @return true if this machine is running some version of Linux 
     */
    public static boolean isLinux()
    {
        return OS_String.contains("Lin");
    }
    
    /**
     * Contains the per core L2 Cache size. The value returned will be '0' if there was an error obtaining the size
     */
    public final static int L2CacheSize;
    static
    {
        if(isWindows())
        {
            String output = null;
            try
            {
                //On windows, the comand line tool WMIC is used, see http://msdn.microsoft.com/en-us/library/aa394531(v=vs.85).aspx 
                
                Process pr = Runtime.getRuntime().exec("wmic cpu get L2CacheSize, NumberOfCores");
                /*
                 * Will print out the total L2 Cache for each CPU, and the number of cores - something like this (2 CPUs) 
                 * L2CacheSize  NumberOfCores
                 * 1024         4
                 * 1024         4
                 */
                
                BufferedReader br = new BufferedReader(new InputStreamReader(pr.getInputStream()));
                StringBuilder sb = new StringBuilder();
                String line = null;
                while( (line = br.readLine()) != null)
                    sb.append(line).append("\n");
                
                output = sb.toString();
            }
            catch (IOException ex)
            {
                Logger.getLogger(SystemInfo.class.getName()).log(Level.SEVERE, null, ex);
            }
            
            output = output.replaceAll("L2CacheSize\\s+NumberOfCores", "").trim();//Remove header
            output = output.substring(0, output.indexOf("\n")).trim();//Get first line
            String[] vals = output.split("\\s+");//Seperate into 2 seperate numbers, first is total L2 cahce, 2nd is # CPU cores
            L2CacheSize = (Integer.valueOf(vals[0]) / Integer.valueOf(vals[1]))*1024 ; //the value is in KB, we want it in bytes
        }
        else if(isMac() || isLinux())
        {
            String output = null;
            try
            {
                //Nix, use /proc/cpuinfo
                Process pr = Runtime.getRuntime().exec("cat /proc/cpuinfo");
                
                BufferedReader br = new BufferedReader(new InputStreamReader(pr.getInputStream()));

                String line = null;
                while( (line = br.readLine()) != null)
                    if(line.startsWith("cache size") && output == null)//We just need one line that says "cache size" 
                        output = line;
                
                
            }
            catch (IOException ex)
            {
                Logger.getLogger(SystemInfo.class.getName()).log(Level.SEVERE, null, ex);
            }
            
            output = output.substring(output.indexOf(":")+1);
            String[] vals = output.trim().split(" ");
            int size = Integer.parseInt(vals[0]);
            if(vals[1].equals("KB"))
                size*=1024;
            else if(vals[1].equals("MB"))
                size*=1024*1024;
            
            L2CacheSize = size;
        }
        else//We dont know what we are running on. 
            L2CacheSize = 0;//TODO is there a way to approximate this? 
        
    }
}

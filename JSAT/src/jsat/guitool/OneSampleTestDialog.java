
package jsat.guitool;

import java.awt.BorderLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.List;
import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;
import javax.swing.JTextField;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import jsat.linear.Vec;
import jsat.testing.onesample.OneSampleTest;
import jsat.testing.StatisticTest.H1;

/**
 *
 * @author Edward Raff 
 */
public class OneSampleTestDialog extends JFrame
{
    final OneSampleTest test;
    final String[] titles;
    final List<Vec> data;
    
    protected JTextField pValueField;

    public OneSampleTestDialog(final OneSampleTest test, final String[] titles, final List<Vec> data)
    {
        super(test.testName());
        this.test = test;
        this.titles = titles;
        this.data = data;
        
        getContentPane().setLayout(new BorderLayout());
        
        pValueField = new JTextField("0");
        pValueField.setEditable(false); 
        
        final JComboBox jcb = new JComboBox(test.validAlternate());
        test.setAltHypothesis((H1) jcb.getSelectedItem()); 
        jcb.addActionListener(new ActionListener() {

            public void actionPerformed(ActionEvent ae)
            {
                test.setAltHypothesis( (H1) jcb.getSelectedItem()); 
                try
                {
                    pValueField.setText(Double.toString(test.pValue())); 
                }
                catch(Exception ex)
                {
                    
                }
            }
        });
        
        final JTextField altHypoFeild = new JTextField("0.0");
        altHypoFeild.getDocument().addDocumentListener(new DocumentListener() {

            public void insertUpdate(DocumentEvent de)
            {
                update();
            }

            public void removeUpdate(DocumentEvent de)
            {
                update();
            }

            public void changedUpdate(DocumentEvent de)
            {
                update();
            }
            
            private void update()
            {
                try
                {
                    double d = Double.parseDouble( altHypoFeild.getText() );
                    test.setAltVar(d);
                    pValueField.setText(Double.toString(test.pValue()));
                }
                catch(Exception ex)
                {
                    
                }
            }
            
        });

        
        
        JPanel hypothesisPanel = new JPanel();
        
        hypothesisPanel.add(new JLabel(test.getNullVar()));
        hypothesisPanel.add(jcb);
        hypothesisPanel.add(new JLabel(test.getAltVar() + ":"));
        hypothesisPanel.add(altHypoFeild);
        
        hypothesisPanel.setBorder(BorderFactory.createTitledBorder("Hypothesis"));
        
        getContentPane().add(hypothesisPanel, BorderLayout.NORTH); 
        
        
        //Tabbed pane for setting the parameters via summary statistics or the selection fo a dataset
        JTabbedPane paremeters = new JTabbedPane();
        
        //First the Summary Statistics
        
        final String[] feildNames = test.getTestVars();
        final JTextField[] feilds = new JTextField[feildNames.length];
        
        
        JPanel statsPanel = new JPanel(); 
        statsPanel.setLayout(new BoxLayout(statsPanel, BoxLayout.Y_AXIS));
        
        for(int i = 0; i < feildNames.length; i++)
        {
            JPanel tmp = new JPanel();
            tmp.add(new JLabel(feildNames[i] + ": ")); 
            feilds[i] = new JTextField("1.0");
            feilds[i].getDocument().addDocumentListener(new DocumentListener() {

                public void insertUpdate(DocumentEvent de)
                {
                    update();
                }

                public void removeUpdate(DocumentEvent de)
                {
                    update();
                }

                public void changedUpdate(DocumentEvent de)
                {
                    update();
                }
                
                private void update()
                {
                    double[] vals = new double[feilds.length];
                    try
                    {
                        for (int i = 0; i < vals.length; i++)
                            vals[i] = Double.parseDouble(feilds[i].getText());
                        test.setTestVars(vals);
                        pValueField.setText(Double.toString(test.pValue())); 
                    }
                    catch (Exception ex)
                    {
                    }
                }
            });
            tmp.add(feilds[i]);
            statsPanel.add(tmp);
        }
        
        
        paremeters.add("Stats", statsPanel);
        
        
        ///Now the Set using data tab
        JPanel datasetPanel = new JPanel();
        
        if(titles != null && titles.length > 0)//The data may not be loaded, but they know the statitics
        {
            final JComboBox dataCB = new JComboBox(titles);


            dataCB.addActionListener(new ActionListener() {

                public void actionPerformed(ActionEvent ae)
                {
                    try
                    {
                        test.setTestUsingData(data.get(dataCB.getSelectedIndex())); 
                        pValueField.setText(Double.toString(test.pValue()));
                    }
                    catch(Exception ex)
                    {

                    }

                }
            }); 


            dataCB.setBorder(BorderFactory.createTitledBorder("Select Data Set:"));
            datasetPanel.add(dataCB);
        }
        else//Well just let them know they need to load some data first
        {
            JTextField jtf = new JTextField("You must have loaded some\ndata set to use this feature");
            jtf.setEditable(false);
           datasetPanel.add(jtf);
        }
        paremeters.add("Data", datasetPanel);
        
        
        getContentPane().add(paremeters, BorderLayout.CENTER);
        
        JPanel pValPanel = new JPanel();
        pValPanel.setBorder(BorderFactory.createTitledBorder("P Value")); 
        pValPanel.add(pValueField); 
        
        getContentPane().add(pValPanel, BorderLayout.SOUTH);
        
    }
    
    
}

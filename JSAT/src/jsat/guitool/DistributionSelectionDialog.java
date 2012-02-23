package jsat.guitool;

import java.awt.BorderLayout;
import java.awt.Frame;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JDialog;
import javax.swing.JFormattedTextField;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextField;
import jsat.distributions.Distribution;

/**
 *
 * @author Edward Raff
 */
public class DistributionSelectionDialog extends JDialog
{
     Distribution[] distributions;

     final JPanel variablePanel;
     final JComboBox jc;
     String[] vars;


    public DistributionSelectionDialog(Frame owner, String title, Distribution[] distributions)
    {
        super(owner, title, true);
        this.distributions = distributions;

        JPanel panel = new JPanel(new BorderLayout());
        variablePanel = new JPanel(new GridLayout(1, 1));
        jc = new JComboBox(distributions);
        jc.addActionListener( new ActionListener() {

            public void actionPerformed(ActionEvent e)
            {
                Distribution dist = (Distribution) jc.getSelectedItem();

                vars = dist.getVariables();
                double[] defaultVals = dist.getCurrentVariableValues();
                variablePanel.removeAll();
                variablePanel.setLayout(new GridLayout(vars.length, 1));
                for(int i =0; i < vars.length; i++)
                {
                    String var = vars[i];
                    JTextField jt = new JFormattedTextField(defaultVals[i]);
                    jt.setBorder(BorderFactory.createTitledBorder(var));
                    variablePanel.add(jt);
                }
                variablePanel.revalidate();
            }
        });

        jc.setSelectedIndex(0);

        JButton okButton = new JButton("OK");
        okButton.addActionListener(new ActionListener() {

            public void actionPerformed(ActionEvent e)
            {
                setVisible(false);
            }
        });

        panel.add(jc, BorderLayout.NORTH);
        panel.add(new JScrollPane(variablePanel), BorderLayout.CENTER);
        panel.add(okButton, BorderLayout.SOUTH);

        this.getContentPane().setLayout(new GridLayout(1, 1));
        this.getContentPane().add(panel);
    }

    public Distribution getDistribution()
    {
        setSize(300, 300);
        setVisible(true);

        if(vars == null)//Cancled
            return null;

        int index = jc.getSelectedIndex();

        for(int i = 0; i < variablePanel.getComponentCount(); i++)
        {
            distributions[index].setVariable(vars[i], Double.parseDouble(((JFormattedTextField)variablePanel.getComponent(i)).getText()));
        }

        return distributions[index].clone();
    }

}

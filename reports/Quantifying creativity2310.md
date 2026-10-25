 

**MEASURING CREATIVITY AS IT UNFOLDS IN INTERDEPENDENT TEAMS** 

**Felix Mosner & Matthias Lehmann**

**Motivation & Introduction**  
Creativity has long been recognized as the starting point of innovation and a key ingredient for firm survival and renewal (Amabile, 1988; Anderson, Potočnik, & Zhou, 2014). It is commonly defined as the production of ideas or actions that are both novel and useful (Amabile, 1996; Shalley, Zhou, & Oldham, 2004). In organizations, creativity rarely arises in isolation. It results from the interplay of person and context (Woodman, Sawyer, & Griffin, 1993), and much of it emerges in teams, where it depends on how creative contributions are distributed among members, how members build on each other’s contributions, and how these contributions unfold over time (Pirola-Merlo & Mann, 2004; Hargadon & Bechky, 2006; Singh & Fleming, 2010; Harvey, 2014). Yet our measures of team creativity still struggle to accurately capture this process. They assess creativity ex-post, oftentimes through supervisor evaluations or implemented ideas, rather than observing how creative contributions emerge as the work unfolds. As a result, the process behind a creative outcome often remains hidden: the options a person had but did not take, the preparatory actions by teammates that made an idea possible, and the many creative attempts that never reached evaluation. What we call a creative contribution is therefore often only its final, visible step.  
In line with the scope of this special issue, this proposal addresses that gap by using artificial intelligence (AI) to measure creativity as it occurs, that is, as creative behavior unfolds naturally and in real time in interdependent teams. Analyzing more than half a million on-ball decisions from 418 professional matches and evaluating each against the 1,771 actions the player could have taken instead (nearly one billion alternatives in total), we use a longitudinal approach and study two outcomes: whether an individual uses the creative opportunity available in a situation, and how an action changes the creative opportunities of a team member who acts next. Our independent variables are the creative opportunity itself, the preceding actions of team members, and critical events such as goals, red cards, and tactical changes. We aim to make three contributions.   
First, we develop a new measure for creativity that assesses every single action of every team member on two dimensions, its surprise (i.e. novelty in context) and its usefulness, which can be transferred to the digital trace data organizations increasingly generate and analyze (Yang et al., 2022). Second, we provide fine-grained evidence on how creativity is distributed within teams and how it shifts in response to performance feedback or changes in team composition, such as early success, the loss of a member, or managerial reconfigurations. Using complete behavioral records from professional football, we shed light on the build-up process of creative outcomes and make visible creative contributions that conventional evaluation overlooks: those that prepare a success rather than complete it, much like the work that leads up to an idea being accepted for implementation. Third, we contribute to the debate about human and AI creativity and propose a perspective where AI could help organizations recognize hidden creative contributions and the process through which creativity builds up within teams.

**Current Measures Miss the Creative Process**  
Organizational research has captured creativity mainly through supervisor ratings (Oldham & Cummings, 1996; Zhou & George, 2001), self-reports, which often diverge from other-rated measures (Ng & Feldman, 2012), outcome proxies such as patents (Fleming, 2001; Audia & Goncalo, 2007), contributions to idea contests and crowdsourcing platforms (Girotra, Terwiesch, & Ulrich, 2010; Bayus, 2013), or the recognition and market success of creative products (Cattani & Ferriani, 2008; Berg, 2016). These measures try to limit bias by relying on external assessors or observable outcomes, but they have important limitations. These measures judge creativity after the fact. Raters see outcomes rather than choices, tend to penalize novelty (Boudreau et al., 2016; Lane et al., 2022), and outcome proxies capture only ideas that survived selection (Audia & Goncalo, 2007). They cannot sufficiently unravel the emergence of a creative moment or show who in a team contributes creatively at what time, or how this pattern shifts when the team faces pressure or failure.

**AI as a Lens on Human Creativity**  
AI has become a central topic in creativity research, but mostly as creator or collaborator. Generative AI produces ideas that compete with those of humans (Boussioux et al., 2024\) and augments employee creativity (Jia et al., 2024), while homogenizing ideas at the collective level (Doshi & Hauser, 2024). Far less attention has gone to AI as a measurement instrument, although management scholars increasingly use machine learning to build theory from behavioral data (Shrestha et al., 2021; Srivastava et al., 2018). AI can provide what no human rater can: a calibrated estimate of the options an actor had in a situation that never repeats, and a valuation of each option before its outcome is known. Existing automated measures score lab tasks or patents (Organisciak et al., 2023; Funk & Owen-Smith, 2017), not creative acts inside teams.

**Our Approach: Surprise × Usefulness**  
***Setting.*** Professional football offers a rare setting for our questions. Like many organizational teams, football teams combine interdependent roles, a shared goal under time constraints, continuous performance feedback, and changes in membership and strategy during the task (Day, Gordon, & Fink, 2012; Fonti, Ross, & Aversa, 2023), and organizational studies have used sports to study team decisions under performance feedback (Lehman et al., 2011; Lehman & Hahn, 2013). Modern event data from professional football provides a unique opportunity to start disentangling the creative process as it unfolds. Every on-ball action is recorded and coded, the position of every player and the ball in a given moment, and whether it was a throw-in, a shot on goal or something else. With this data, AI models can learn what is typical and what is valuable in a given situation (Decroos et al., 2019; Wang et al., 2024), which is needed to judge novelty and usefulness of a particular action. 

***Measure***. We operationalize novelty in context: an action is novel to the extent that it was improbable in its situation, given what players typically do there, which corresponds to the originality criterion in Simonton's (2012) definition of creativity. We refer to this improbability as surprise. Using StatsBomb open data on 536,223 on-ball decisions from 418 matches of men's and women's World Cups, European Championships, and selected club seasons, we train two models. A graph neural network (*surprise model*) estimates the probability of each of 1,771 possible actions (type, target zone, and pass height), given the positions and movements of all visible players, the ball, the preceding actions, the score, and the time. A second model (*usefulness model*) estimates each action's usefulness, its contribution to the probability of scoring and not conceding, and performs on par with established valuation models.  
Our primary measure scores how surprising the chosen action is among the options that would have improved the team's position; actions without such value score zero. It thus separates creative actions (surprising and useful) from routine and competent actions (expected) and from failed gambles (surprising but not useful). The same models describe each situation's creative opportunity: the options that would have been both unusual and valuable. It produces a creativity score for every action, player, team, and moment of a match, which opens up a whole stream of possibilities to analyze team creativity as it unfolds, changes and crumbles. 

**What We Have Done**  
We have implemented and validated a first version of the full pipeline on the 536,223 on-ball decisions in 418 matches. All scores are out-of-sample, as each decision is scored by models that never saw its match. The surprise model is well calibrated across competitions and for men's and women's football, and the usefulness model matches established valuation models. As a post-hoc validation, we find players known for their creativity to score higher than other players in the same position (AUC \= 0.65, 95% CI \[0.57, 0.73\]), clearly more so than on usefulness alone. The measure is largely independent of position and singles out through balls and passes that bypass opponents. First substantive results show that creative build-up play predicts shots and goals, and that team creativity drops after a team scores or receives a red card but rises after tactical changes. Our analysis also reveals creative actions, that would usually remain hidden. For instance, in possessions that end in a goal, creativity peaks at the assist and the pass before it (about \+0.5 SD), highlighting that an implemented idea could sometimes just be the pinnacle of creative actions. 

**What We Will Do**  
***Enabling creativity within teams.*** We will measure how each action changes the creative opportunities of the teammate who acts next and identify members whose contribution lies in creating room for others rather than in visible creative acts. This allows us to test whether teams benefit more from creativity concentrated in a few stars or from enabling contributions distributed across members (Singh & Fleming, 2010; Groysberg, Polzer, & Elfenbein, 2011). First results suggest that concentration alone does not predict performance.

***Creative opportunity and its use.*** We will compute the creative opportunity in each situation and test when members use it, comparing roles and following each team through a match. Changes in the team’s situation serve as natural tests: after a goal, a red card, or a tactical change, the room for creativity may stay the same while the willingness to use it shifts. Our first results show lower creativity after scoring and after red cards. The distinction matters here: success may leave opportunities unused, in line with evidence that past success narrows creative search (Audia & Goncalo, 2007), whereas losing a member may shrink the opportunities themselves. For organizations, these moments correspond to a won client, a departing member, or a manager’s restructuring.

**Potential Risks**  
Two organizational risks could limit how far our vision reaches. The first concerns access to data. We clearly see settings that could provide the record of decisions in context, a model of typical behavior, and an observable outcome, such as shared code repositories, digital team interactions or collaboration data from platforms. However, the willingness of companies to share that data is hard to determine, as it is governed by privacy law and employee representation. The second risk concerns misuse and acceptance. A measure that scores every action could become a monitoring tool, and people who know they are scored may produce surprise for its own sake. Managers might also reject an algorithmic judgment of creativity or trust it too much. We therefore position the measure as a team-level diagnostic instrument rather than an individual performance metric, and will develop guidelines for its responsible use.

**How Organizational and Social Science Would Change**  
This project changes how creativity can be studied: instead of rater-based snapshots, researchers can follow creativity as a process. Instead of judging creativity by its final, visible step, they can see how it builds up: which opportunities a team member had, whether they used them, and how one member’s action opened room for the next. This allows theories of team creativity, so far tested on aggregated outcomes, to be examined at the level of individual actions, as teams respond to performance feedback and changes in team composition.   
For organizations, the measure also allows to detect hidden creative contributions. Evaluations tend to credit visible outcomes and the people who deliver them, while preparatory work stays invisible (Star & Strauss, 1999\) and novelty is sometimes penalized (Boudreau et al., 2016). By judging each action against the options its author had, the measure recognizes members who create room for others. It also distinguishes missing from unused opportunities, which call for different responses: changes in structure in the first case, motivation and psychological safety (Edmondson, 1999\) in the second.  
The reach of this approach extends well beyond football. As work increasingly runs through digital tools, organizations generate a growing stream of trace data (Leonardi, 2021). Microsoft, for instance, studied how remote work changed collaboration using anonymized email, calendar, chat, and call records of more than 60,000 employees (Yang et al., 2022), and team communication on collaboration platforms has been used to explain team performance (Lix et al., 2022). These can be analyzed now in new and innovative ways as language models now provide a model of typical behavior that can be compared to actual behavior. Our approach offers a template for doing so and positions AI not as a creator of ideas, but as an instrument that makes human creativity visible.

**![][image1]**

![][image2]

**References**

Amabile, T. M. (1988). A model of creativity and innovation in organizations. *Research in Organizational Behavior, 10*, 123–167.

Amabile, T. M. (1996). *Creativity in context*. Westview Press.

Anderson, N., Potočnik, K., & Zhou, J. (2014). Innovation and creativity in organizations: A state-of-the-science review, prospective commentary, and guiding framework. *Journal of Management, 40*(5), 1297–1333.

Audia, P. G., & Goncalo, J. A. (2007). Past success and creativity over time: A study of inventors in the hard disk drive industry. *Management Science, 53*(1), 1–15.

Bayus, B. L. (2013). Crowdsourcing new product ideas over time: An analysis of the Dell IdeaStorm community. *Management Science, 59*(1), 226–244.

Berg, J. M. (2016). Balancing on the creative highwire: Forecasting the success of novel ideas in organizations. *Administrative Science Quarterly, 61*(3), 433–468.

Boudreau, K. J., Guinan, E. C., Lakhani, K. R., & Riedl, C. (2016). Looking across and looking beyond the knowledge frontier: Intellectual distance, novelty, and resource allocation in science. *Management Science, 62*(10), 2765–2783.

Boussioux, L., Lane, J. N., Zhang, M., Jacimovic, V., & Lakhani, K. R. (2024). The crowdless future? Generative AI and creative problem-solving. *Organization Science, 35*(5), 1589–1607.

Cattani, G., & Ferriani, S. (2008). A core/periphery perspective on individual creative performance: Social networks and cinematic achievements in the Hollywood film industry. *Organization Science, 19*(6), 824–844.

Day, D. V., Gordon, S., & Fink, C. (2012). The sporting life: Exploring organizations through the lens of sport. *Academy of Management Annals, 6*(1), 397–433.

Decroos, T., Bransen, L., Van Haaren, J., & Davis, J. (2019). Actions speak louder than goals: Valuing player actions in soccer. In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining* (pp. 1851–1861).

Doshi, A. R., & Hauser, O. P. (2024). Generative AI enhances individual creativity but reduces the collective diversity of novel content. *Science Advances, 10*(28), eadn5290.

Edmondson, A. C. (1999). Psychological safety and learning behavior in work teams. *Administrative Science Quarterly*, 44(2), 350–383. 

Fleming, L. (2001). Recombinant uncertainty in technological search. *Management Science, 47*(1), 117–132.

Fonti, F., Ross, J.-M., & Aversa, P. (2023). Using sports data to advance management research: A review and a guide for future studies. *Journal of Management, 49*(1), 325–362.

Funk, R. J., & Owen-Smith, J. (2017). A dynamic network measure of technological change. *Management Science, 63*(3), 791–817.

Girotra, K., Terwiesch, C., & Ulrich, K. T. (2010). Idea generation and the quality of the best idea. *Management Science, 56*(4), 591–605.

Groysberg, B., Polzer, J. T., & Elfenbein, H. A. (2011). Too many cooks spoil the broth: How high-status individuals decrease group effectiveness. *Organization Science, 22*(3), 722–737.

Hargadon, A. B., & Bechky, B. A. (2006). When collections of creatives become creative collectives: A field study of problem solving at work. *Organization Science, 17*(4), 484–500.

Harvey, S. (2014). Creative synthesis: Exploring the process of extraordinary group creativity. *Academy of Management Review, 39*(3), 324–343.

Jia, N., Luo, X., Fang, Z., & Liao, C. (2024). When and how artificial intelligence augments employee creativity. *Academy of Management Journal, 67*(1), 5–32.

Lane, J. N., Teplitskiy, M., Gray, G., Ranu, H., Menietti, M., Guinan, E. C., & Lakhani, K. R. (2022). Conservatism gets funded? A field experiment on the role of negative information in novel project evaluation. *Management Science, 68*(6), 4478–4495.

Lehman, D. W., & Hahn, J. (2013). Momentum and organizational risk taking: Evidence from the National Football League. *Management Science, 59*(4), 852–868.

Lehman, D. W., Hahn, J., Ramanujam, R., & Alge, B. J. (2011). The dynamics of the performance–risk relationship within a performance period: The moderating role of deadline proximity. *Organization Science, 22*(6), 1613–1630.

Leonardi, P. M. (2021). COVID-19 and the new technologies of organizing: Digital exhaust, digital footprints, and artificial intelligence in the wake of remote work. *Journal of Management Studies, 58*(1), 249–253.

Lix, K., Goldberg, A., Srivastava, S. B., & Valentine, M. A. (2022). Aligning differences: Discursive diversity and team performance. *Management Science, 68*(11), 8430–8448.

Ng, T. W. H., & Feldman, D. C. (2012). A comparison of self-ratings and non-self-report measures of employee creativity. *Human Relations, 65*(8), 1021–1047.

Oldham, G. R., & Cummings, A. (1996). Employee creativity: Personal and contextual factors at work. *Academy of Management Journal, 39*(3), 607–634.

Organisciak, P., Acar, S., Dumas, D., & Berthiaume, K. (2023). Beyond semantic distance: Automated scoring of divergent thinking greatly improves with large language models. *Thinking Skills and Creativity, 49*, 101356\.

Pirola-Merlo, A., & Mann, L. (2004). The relationship between individual creativity and team creativity: Aggregating across people and time. *Journal of Organizational Behavior, 25*(2), 235–257.

Shalley, C. E., Zhou, J., & Oldham, G. R. (2004). The effects of personal and contextual characteristics on creativity: Where should we go from here? *Journal of Management, 30*(6), 933–958.

Shrestha, Y. R., He, V. F., Puranam, P., & von Krogh, G. (2021). Algorithm supported induction for building theory: How can we use prediction models to theorize? *Organization Science, 32*(3), 856–880.

Simonton, D. K. (2012). Taking the U.S. Patent Office criteria seriously: A quantitative three-criterion creativity definition and its implications. *Creativity Research Journal, 24*(2–3), 97–106.

Singh, J., & Fleming, L. (2010). Lone inventors as sources of breakthroughs: Myth or reality? *Management Science, 56*(1), 41–56.

Srivastava, S. B., Goldberg, A., Manian, V. G., & Potts, C. (2018). Enculturation trajectories: Language, cultural adaptation, and individual outcomes in organizations. *Management Science, 64*(3), 1348–1364.

Star, S. L., & Strauss, A. (1999). Layers of silence, arenas of voice: The ecology of visible and invisible work. *Computer Supported Cooperative Work*, 8(1–2), 9–30.

Wang, Z., Veličković, P., Hennes, D., Tomašev, N., Prince, L., Kaisers, M., … Tuyls, K. (2024). TacticAI: An AI assistant for football tactics. *Nature Communications, 15*, 1906\.

Woodman, R. W., Sawyer, J. E., & Griffin, R. W. (1993). Toward a theory of organizational creativity. *Academy of Management Review*, 18(2), 293–321.

Yang, L., Holtz, D., Jaffe, S., Suri, S., Sinha, S., Weston, J., Joyce, C., Shah, N., Sherman, K., Hecht, B., & Teevan, J. (2022). The effects of remote work on collaboration among information workers. *Nature Human Behaviour, 6*(1), 43–54.

Zhou, J., & George, J. M. (2001). When job dissatisfaction leads to creativity: Encouraging the expression of voice. *Academy of Management Journal, 44*(4), 682–696.
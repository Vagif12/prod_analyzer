import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import HybridFusion
import os
from semantic_router.encoders import OpenAIEncoder
import semchunk

encoder = OpenAIEncoder()

from semantic_chunkers import StatisticalChunker
chunker = StatisticalChunker(encoder=encoder)

from langchain_text_splitters import RecursiveCharacterTextSplitter


splitter2 = RecursiveCharacterTextSplitter(chunk_size=1000)


wcd_url = os.environ["WEAVIATE_URL"]
wcd_api_key = os.environ["WEAVIATE_KEY"]

connection = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,  # Replace with your Weaviate Cloud URL
    auth_credentials=Auth.api_key(wcd_api_key),  # Replace with your Weaviate Cloud key
    headers={'X-OpenAI-Api-key': os.environ["OPENAI_API_KEY"]})


connection.collections.create(
    "Content",
    vectorizer_config=Configure.Vectorizer.text2vec_openai()
)
collection = connection.collections.get("Content") 


text = """
PRODUCTION SUMMARY


By: Lia Kondova
Year: 2021

Note: POSSIBLE EXAM QUESTIONS are in RED


Table of Contents


Lecture 0: Introduction	3
Lecture 1: Materials	4
Lecture 2.1: Casting - General	6
Lecture 2.2: Casting – Expendable molds	8
Lecture 2.3: Casting – Permanent molds	9
Lecture 2.4: Casting – Design rules	11
Lecture 3.1: Shaping – 3D deformation	13
Lecture 3.2: Shaping – Sheet metal	16
Lecture 3.3: Shaping – Sheet metal forming	18
Lecture 4.1: Separating - Mechanical	22
Lecture 4.2: Separating - Physical	25
Lecture 5: Machining	28
Lecture 6: Joining	36
Lecture 7: Plastic processing	43
Lecture 8: Additive processes	46
Appendix: Surface finish roughness of various manufacturing processes	49



Lecture 0: Introduction
Production – the process of making goods to be sold

Production 1 – translating design ideas into manufacturing products
Pushing the limits of:
Time – how much time do we take? how much products can we produce per hour?

Quality – if we’re faster, will we lose quality? – for some parts yes, for others no

Cost – invest in tooling, will it be earned back if we make enough products?
Some processes are not used if the batch time is too small

technical abilities – some processes in some materials easily work, but in other materials they won’t work

logistic capabilities – does it make sense to make one machine faster, if it will make production slower? when does it make sense to optimize?

Goal – integrate manufacturing, as an inherent part of the design cycle – what if we have to make a thousand or a million products, will it change the way we think about the geometry and processes?
Function - Designer focuses on the function of the product – what we intend the product to do
Function-Shape - Function is related to shape – does shape appeal to the users? can this shape realize this functionality?
Function-Material - some functionality can’t be realized by some types of material – we can’t make a rubber hammer
Shape-Process – many processes have limitations in terms of what shapes you can make with them; you can’t make every shape with every process
Production-Material – some processes only work with materials that are conductive, tough, etc.
Process-Material – if we want to have tiny products, we need a specific material, we can’t make every material into every shape



Lecture 1: Materials
Material – consists of elementary particles, which directly drive what is possible in production (some materials are suitable only for some processes)
Atomic structure – atoms cluster with other atoms of the same type
E.g. – metals - consist of many little crystals, oriented randomly
not every set of atoms is a material

Molecules – atoms connect to other atoms
E.g. - plastics

Properties are determined by:
The types of atoms
The type of connection between the atoms

Metals – have crystal structure
Atoms attempt to fill the space as effectively as possible
Different types of atoms attempt that in different ways
Body-centered cubic (bcc) – the atoms try to find the most energetic optimal spot, given the atoms of the previous layer that it rests on
one layer of atoms, then the atoms of the second layer rest between each two atoms at the previous layer, third layer is strictly above the first layer
the easier to push the atoms around 🡺 the easier the material will be by default

Face-centered cubic (fcc)
e.g. – tungsten, iron under 914 degrees Celsius

Hexagonal close-packed (hcp) – the first layer is not so nicely stacked, then the atoms of the second layer are positioned between each two atoms of the first layer
hexagonal shape 🡺 difficult to deform
e.g. – titanium, zinc







Crystal structure – atoms connect to other atoms of the same type

Metal
When we solidify metal (heating it up, then cooling it down), grains are formed
It cools down fast 🡺 material won’t have time to form bigger crystals
the more grains 🡺 the smaller the grains 🡺 harder and stronger material 🡺 more difficult to deform

It cools down slowly 🡺 bigger crystals
Risk of segregation – the different types of atoms in a metal have different melting points 🡺if we then cool it down very slowly, first the material with the highest melting point will solidify & sink to the bottom, then the second layer of atoms will be liquid 🡺 the two types will be separated in different layers, instead of mixed

Anisotropic behavior – the value of material properties changes with the direction of the measurement
A material is subjected to cold deformation (the material doesn’t recrystallize) – its volume doesn’t change, but the grains deform 🡺 this new irregular structure has a preferred direction 🡺 the material behavior changes with the direction in which you measure it

Plastics = polymers + a lot of additions
Polymer – very large molecule
has a repetitive pattern – the many chunks of polymer are connected to a chain and the chain is interwoven

additions
cheaper filler (polymers are expensive)
improves mechanical properties – add glass/metal/etc. to make material harder
improves process ability – change the viscous temperature of the material
the more we mix plastics 🡺 the more additives we add 🡺 the more difficult it will become for recycle

Types of plastic
Thermoplastic material – heat plastic 🡺 it will become viscous (never liquid)🡺 easier to deform

Elastomers – combination between thermosetting and thermoplastic materials
Rubbers

Thermosetting material – heat this material 🡺 itdisintegrates
Easier to make UV-resistant

Co-polymer - macromolecule from different monomers
One molecule consists of different parts
E.g. – spaghetti – each spaghett has different a color

Poly-blends - different types of macro-molecules
E.g. – spaghetti – each spaghett has its own color, different from the rest

Lecture 2.1: Casting - General
Casting - turning shapeless raw material into a usable shape for the first time
Forming – to begin, to exist, to make something new

Shaping – to make something become a particular shape
Near-net-shape production – we immediately go from the raw material to the almost finished product

Material type
Liquid 🡪 casting
Pour the molten material into a mold, let cool down
E.g. – metal casting

Viscous 🡪 plastic processing
we need a lot of pressure
E.g. – plastics

Powder 🡪 powder metallurgy (sintering)
Materials with very high melting point 🡺 we start with powder, push it really hard together, then bake in together

Types of molds
Expendable molds – made from sand/ceramics; the mold will be damaged in the process 🡺 has to be made over and over for every product
Sand casting
shell molding
investment casting (low-wax process)

Permanent molds – made from metal; for products needed in large quantities (pens) 🡺 has to be done in a way so the product can be removed from the mold
Permanent-mold casting
Die casting
Centrifugal casting

Composite molds – combination of expendable and permanent

Metal casting
Complex shapes
Large products in one piece
Mostly cheap 🡺 mass production, small quantities
Isotropic final products – the product will behave the same in all directions (unlike anisotropic products) because when the metal starts to cool down, the grains will start to grow at different positions towards each other 🡺 no deformation

Economic batch size
Single piece - church bell; 1 mold that takes a long time to make
hundreds of thousands pieces - pens; we make one mold that is used over and over
Flow of molten metal
Supply
Distribution

Solidification and cooling – the volume of the metal decreases as it cools down, so a bit more has to be poured in the mold in case extra material is needed
Heat transfer
Shrinkage - the measures of a wooden model differ from those on the construction drawing 🡺 we need specific rulers that are slightly off the nominal value 🡺 this helps to have the exact size of the material when it cools down
Exam question: What would be the true size of 1000 mm (= 1m) on a special ruler for aluminum if it has a volumetric shrinkage of 6.6%
Answer: volumetric shrinkage is always 3 dimensional
l*w*h - 6.6% = 1 m3
assume l=w=h=y
y3 – 0.066 * y3 = 1
y3 * (1-0.066) = 1 
y3 = 1/(1-0.066)
y (length) = 1.023 m 🡺 1000 mm on the ruler for aluminum is: 1023 mm

Removal – issue because when you try to lift the mold, it creates vacuum with the sand that it was cast into and damages it
Solution: Drafting angle – create angle in the sand so when you lift the mold air goes it immediately

Defects
Cavities – the mold is initially filled with air and we have to push it out to escape, but an error occurs and the air cannot escape and it is trapped inside 🡺 damage in the product
Blowholes – large spherical holes near the surfaces
Shrinkage cavities - pipe at the top caused by shrinkage during the solidification 

Discontinuities – if we have very complex products, the material might be poured from two sides 🡺 there might not be complete fusion
cold shut - interferance in a casting because the metal was poured from two sides
cracks – caused by a temperature difference between the metal poured from two sides
cold and hot tears
hot tearing - coarse grain size and the presence of low-melting-point segregates along the grain boundaries
inclusion -  reduction in the strength of the material caused by the way of casting

Lecture 2.2: Casting – Expendable molds
Expendable molds – destroyed when removing the product 🡺 has to be remade every time
Sand casting – limited to metals 
No limit to size/weight/shape
Low tooling cost – the pattern is made of wood
Relatively coarse surface finish
Product geometry - all kinds of curved shapes; large products
rough surface because of the roughness of the sand
Product material – almost all materials
Mold material
Naturally bonded sand
Synthetic bounded sand
Silica sand

Elements
Pouring cup, sprue and runner - supply of molten metal
Risers – prevent shrinkage
Cores – define interior region

Cores and parting lines
Hollow regions can be formed by internal cores
External cores are used with big or complex models to form features
Parting lines are visible of the product

Shell molding – layer of sand around a complex but rough model with high precision
High accuracy – positioning holes are possible
High production rate
Limited part size - complex process which takes a lot of handling
Expensive equipment – the pattern is not made of wood, but of metal
Product geometry – smooth surfaces, acute angles, thin walls
Mold material – coated sand, hardened in the oven

Investment casting – not only we lose the mold, but also the pattern (jewelry is made in this way)
Very good surface finish – we use ceramics
Limited part size – complex process which takes a lot of handling
Expensive materials and labor
Mold materials – ceramics
we don’t need a draft angle, we make small products because they are complex
Product material – cast iron/steel, titanium alloy, nonferrous metal
Lecture 2.3: Casting – Permanent molds
Permanent molds – reusable molds that can be used to make more than one product
Only possible for shapes that “release” 🡺 undercuts are impossible
Draft angles are essential to prevent mold from being damaged (the wear will be too high)
Can be done under sheer gravity or with some pressure
Lead time – long because it takes more time to prepare the mold
Good surface finish
High accuracy
Limited part complexity – because of a lot of contact between the product and the metal
If parts are complex 🡺 it will be too difficult to remove the parts from the mold
High mold cost – the mold is not a box of sand anymore
Automation – we have such precise metal blocks, they can be positioned by machines 🡺 high production rate
Mold material
Metal
Metal or sand core
Product material
Nonferrous metal
Cast iron

Process
Cooling channels
Ejectors – during cooling down the product might get stuck on the mold 🡺 we need to make sure it falls out before we start making the next product 🡺 ejector pins push out the product as soon as possible
Thermal barrier coating

Vacuum casting – we push the material into the mold and suck the air out so it fills the entire mold
Useful when we don’t want air bubbles into the product

Die casting (pressure die casting)
Hot chamber – the two dies are pushed together, the furnace is immediately connected to the molds, a plunger goes up so that the material can flow into the machine, then the plunger goes down and closes the entrance by pushing down, thus it increases the pressure in the machine, thus the material is pushed into the mold with high pressure, when it cools down the product is pushed out by the ejector pins
Oven is part of the machine 
3-30 MPa pressure
Cooling by water or oil – the mold becomes more expensive, but the production goes at a higher rate

Cold chamber – the metal is pushed into a horizontal cylinder, the
plunger pushes to the left and the material fills the mold
Oven is outside of the machine
70-200 MPa pressure
>70% of the machine consists of the mechanism that closes the dye
Rotational molding – the machine is turning into unpredictable ways, the small granulated pellets start melting and become viscous 🡺 as soon as they melt they stick to the walls of the mold and slowly form the outside of the product, the rest of the gravel will be dispersed unevenly/randomly to form the inside of the wall
Moderate production rate – the molding system is not too fast, but is automated
Relatively cheap molds – the molds are made of many thin components 🡺 relatively easy and cheap to make
Significant infrastructure required – we need a big furnace
Molds can consist of multiple parts – we are not relying on two parts of the mold opening and closing, but on more parts 🡺 more complex products
Mold material
Metal - thin
Product geometry
Hollow parts (even closed) – specific for rotational modeling
No small parts
Product material
Foremost plastics

(Semi)centrifugal casting – if we start rotating the system faster 
High production rate – because of the speed of the rotation (faster with centrifugal casting)
Expensive equipment – because of the speed
Characteristics - no ejector pins, no parting lines, very symmetrical products
Limited part shape – because the molds have to be bigger than the products, as well as to be able to rotate very fast
Mold material
Metal
Graphite
Product geometry
Hollow, cylindrical parts
Limited part shape
Product material
Non-ferrous and ferrous metals and alloys

Surface finish of casting processes



Lecture 2.4: Casting – Design rules
Design rules
Adjust the design to the simplest casting process
Avoid accumulation of metal 
Corners
Wall thickness
Avoid large flat areas
Ribs
Limit stress by shrinkage
Adjust a draft 

Choosing the right process
Technical possibilities/limitations
Product material
Product geometry
Product size
Product complexity
Dimension accuracy
Quality of surface finish
Level of detail

Manufacturing costs
Equipment costs
Labor costs
Production rate
Inititial period
Required finishing

Designing the process
Locate parting line
if there is a really clear parting line 🡺 sand casting/shell casting
thin but very explicit line 🡺 permanent mold casting
more than one parting line 🡺 rotational molding

Locate ejection point/area – indicate (high) pressure casting 🡺 we exclude sand/shell casting
Choose runner
Locate sprues and risers
Required finishing











Casting processes summary


Overview of processes’ batch size, mold material, etc.




Overview of processes’ product shapes


Lecture 3.1: Shaping – 3D deformation
Forming - ‘To begin, to exist or to make something begin to exist’
Shaping - ‘To make something become a particular shape’
Deformation
Hot deformation
temperature that we use to deform is 70-80% of the melting temperature 🡺 reduction of force
the material becomes very hot 🡺 crystals lose their initial shape 🡺 when material starts cooling down 🡺 recrystallization 🡺 improved structure
a lot of energy is used 🡺 not the most efficient

Warm deformation
Material is warm enough to reduce force, but not warm enough for recrystallization
30-40% of melting temperature

Cold deformation
No recrystallization – we deform the crystals that already exist
Work-hardening

Processes
Continuous – go on and on as long as we can provide more raw material
Rolling
Flat
Shape
Extrusion
Drawing

Discrete – we make discrete individual products
Forging
Sheet-metal forming
Powder metallurgy

Forging - shaping by the application of compressive forces 
High equipment and labor costs
Moderate-to-high operator skills
High strength/toughness

Product geometry
Discrete products
Complex shapes
Very small up to very big

Process – batch

Cold forging
Good surface finish and accuracy
Crystal deformation


 Hot forging
Lower forces required
Worse surface finish and accuracy
Strenghtening







Open-die Forging
Limited shapes – difficult to make complex shapes
Small quantities – because everything is done by hand
Mostly hot forging – to reduce the amount of forces
Barreling


Closed-die Forging – the die has the shape information we need
More complex shapes
Large quantities
Flash – excess material that extends outside the shape we want (because the die doesn’t close perfectly)


Heading – changing the shape in a number of steps, each step gets closer to the shape we want 









Coining
high accuracy and surface finish
we don’t have a flash 








Extrusion
Moderate-to-high die and equipment cost
Low-to-moderate labor cost
The material is under high triaxle compressive stresses 🡺 large deformations can take place without fracture
dead-metal zone - the metal at the corners is essentially stationary

Product geometry
Long lengths – we push the material in a long die
Constant cross section
Cut for discrete products

Process - each billet is extruded individually 🡺 extrusion is a batch or semi continuous operation

Extrusion force


Cold extrusion
High ductility
For individual production
Products up to 45kg with lengths up to 2 m
Combined with forging
k = a · Hm

Hot extrusion
Low ductility
For batch production
Heat requirements
k = Cextr(T)



 
Indirect extrusion
No billet-container friction
Hot extrusion

Hydrostatic	
Fluid pressure
Reduced friction

Design – we use extrusion for flat thing, sharp angles/corners are difficult because of the friction between the die and the material
 
Example exam question (trick question) – would you be able to use steel as extrusion material
Answer: forces are extremely high 🡺 the die has to be extremely strong to withhold the force 🡺 we cannot extrude steel because we don’t have a strong enough die

Drawing – we don’t push the material through the die, we pull it; only works when the material is COLD
Low-to-moderate equipment and labor costs
Low-to-moderate operator skills required
Product geometry
Long rods and wires
Different cross sections 
Process
Continuous
Friction force

Lecture 3.2: Shaping – Sheet metal
Sheet metal components have extremely large surface-to-volume ratios 🡺 forces in sheet metal are foremost planar (2D)

Metal rolling
Sheet and shape
Expensive equipment
Good surface finish
Low-to-moderate labor cost

Product geometry
Flat plates and foil
Profiles and rails

Process - semi-continuous

Hot rolling
Course-grained, brittle structure of cast metal becomes wrought structure with finer grain
Outcome – the materials are not anisotropic

Cold rolling
Higher strength and hardness
Better surface finish
Outcome – the materials are anisotropic

Flat rolling 
Friction forces the material through the gap
Relative sliding, due to constant surface speed of the roll
Compromise on maximum friction and slip
Maximum draft possible (μ > tan α)


Example exam question – what determines the max thickness reduction in metal rolling?
Answer – the coefficient of friction, not the rolling diameter




Bending and crown
Reducing roll forces by: 
Using smaller diameter rolls
the smaller the diameter 🡺 the smaller the forces
the smaller the diameter 🡺 the smaller the thickness reduction 
no matter how big the rolls are they will still bend slightly because of the forces 🡺 a sheet will be slightly thicker in the middle 🡺 we can make a role that is not perfectly cylindrical but slightly thicker in the middle

Smaller reductions
Back or front tension to the strip

Camber on rolls / crowned rolls
Rolls with high modulus of elasticity

Sheet metal – the stress normal to sheet metal far less than longitudinal stress 
Anisotropy
Normal

Planar


Lecture 3.3: Shaping – Sheet metal forming
Sheet metal forming
Low costs – sheet of metal is like a sheet of paper 🡺 easy to bend 🡺 cheap machines, equipment
Low skills

Product geometry
Thin walls
Wide shape variety 

Processes
Bending
Strain stress – on the outside
Compressive stress on the inside

Cracking – minimum bend radius
Free bending/air bending
Springback - Due to elastic recovery of the plastically deformed sheet after unloading, causes distortion of part and loss of dimensional accuracy, can be controlled by techniques such as overbending and bottoming of the punch
Negative springback does not occur in air bending

Closed-die bending (V-bending)
Little springback
Can also be negative springback - a condition caused by the nature of the deformation occurring within the sheet metal just when the punch completes the bending operation at the end of the stroke 
Large force





Wiping die bending

Roll bending
Three rolls
Various curvature
Large radii

Roll forming
Continuous rolling
Mass production


Stretch forming
Sheet clamped on sides
Strengthening
Small series
Clamming time
Lost material – because we cut some of the material that we don’t need

Die

Deep drawing
Springback
Only cold forming process
Cold material is bent 🡺 anisotropy
Earing (curling of the edges) – caused by planar anisotropy
blank holder pressure is 0.7− 1.0% of the sum of the yield strength and the ultimate tensile strength of the sheet metal
Too high blank holder force 🡺 increases the punch force and causes the cup wall to tear
Too low blank holder force 🡺 wrinkling of the cup flange will occur
With punch and die 
Bottom and walls of one sheet




Process














Wrinkling – reduced by blankholder

Forces in product material
Underneath the blank holder
Tangential compressive strain
Radial tensile strain
Axial pressure

In the product wall
Axial tensile strength
Tangential tensile strength

Explosive forming
Explosive material generates pressure
The sheet explodes against the wall 


Forming machines
Hammers
Drop hammers
Counterblow hammers
Hydraulic hammers

Presses
Mechanical presses
Screw presses
Hydraulic presses


Lecture 4.1: Separating - Mechanical
Separating – removing material without changing its structure
We want to avoid heating up the material, changing the crystalline structure, damaging the rest of the material
Goal: lose as little material in the process as possible
Avoid making chips or sawdust

Terminology
Punching – removing the inner part 

Blanking – removing 
the outer part




Perforating – making holes on the inside of the product

Notching – make notches

Lancing – material is sliced or cut without producing a slug or separating the workpiece

Slitting – making an indentation in the product 

Mechanical separating
Without chip-forming – there is not enough room to allow for the sheet to deform (🡺 sharp edges, rough materials)
Parting/cutting
Universal tools
Clamp prevents tilting
Shear angle: ε ≈ 12°
Guillotine shear – one big straight blade (🡺 straight lines), machine has no idea about the product geometry
Similar cut - scissors

Punching – the tool perforates the sheet, shape of the hole is determined by the shape of the tool
Process happens within tenths of seconds
Often a combination of punching and forming
Forces in punching
Linear with sheet thickness
Linear with shear strength
Linear with perimeter of the tool geometry – equals the outer line of the punched shape (a.k.a the  circumference of parameter of the tool geometry) 🡺 that determines the punching force:

punching force = (sheet thickness) x (shear strength of material) x (circumference of parameter of the tool geometry)

What is the tool made of when the diameter of the punch is smaller than the sheet thickness?
Strength of the tool has to be much stronger than the sheet, else it will break (rule of thumb)
Similar cut – perforator

Nibbling – using multiple steps to reach the final shape for we want (approximating the shapes we want)

With chip cutting
Exercise:
Assume that the machine costs are € 160.00 /h
				Each tool stroke takes 2 seconds (positioning+punching)
				The special tool costs € 250,00
				The generic tools cost € 20,00 each;
				For the generic tools, a tool change takes 2 seconds
	What is the production quantity that justifies buying a special tool? (NP=number of products=1182)



Sawing – tool doesn’t have information about tool geometry 
Tooth set prevents clamping (at least 3 teeth engaged)




Grinding – teeth are too small almost like grains, they grind away the material, rather than cut it 🡺 friction forces are much higher, temperature rise will be higher 🡺 we can use water for cooling
Exam question: We can use cooling liquid when the speed of the grinding disc is <= 50 m/s. If we have higher revolving speed it is difficult to get the cooling liquid on them 🡺 we accept that the disc will not be cooled, we can go up to 100 m/s speed. 

Physical separation
Flame-cutting, plasma-cutting, laser-cutting
Wire electro discharge machining (wire-EDM)
Waterjet-cutting


Lecture 4.2: Separating - Physical
Physical separating – concentrating energy in order to separate our products 🡺 narrow cuts, heat affected zone
Non-mechanical – independent of mechanical properties, forces, etc. 

Flame cutting (oxyacetylene cutting) – setting fire to the material (burning it away) rather than melting it 
Rapid burning of the material 🡺 process creates damages like the cda grading slug

Contour cutting 🡺 fairly easy

Multi-layer cutting is possible

Cutting very thick materials is possible

Cutting very thin sheets is not preferred - the material will burn and melt away and the holes/contour we are cutting will be less predictable

The burned surface will not be shiny 🡺 high roughness

Source of energy
Burning mix of gasses
Combustion temperature < melting temperature - the temperature at which the material starts burning if we add oxygen is lower than the melting temperature
Oxidation is a source of energy/heat to maintain the process 🡺 might cause rust

Advantages
You can bring the tools for cutting wherever you want, there is not special machinery needed

Disadvantages
Not a subtle process

Plasma cutting – we add a tungsten electrode 🡺 difference in voltage between electrodes and the workpiece 🡺  this creates an ionization of gas 🡺 gas will burn even more fiercely (extreme temperatures up to 30 000 degrees)

Extreme temperatures 🡺 we can cut faster and concentrate the energy better where we want 🡺 we can be more precise 🡺 better surface finish than flame cutting

However!! – now we need a bigger power supply and a bigger machine


Laser cutting (Light Amplification by Stimulated Emission of Radiation) – we use light that has the same frequency  (wavelength), we do that in a concentrated way 🡺 high temperatures (e.g. 10.000 ºC)
We can cut steel up to 30 mm, beyond than it is difficult

We can cut with oxygen 🡺 oxidation, more energy 🡺 we can burn harder and thicker layers of sheet
This process decreases the quality and will introduces rust

We can cut with inert gas (gas that doesn’t contain oxygen, pushes it all away) 🡺 very clear cut without damages

We can reach very narrow cuts 🡺 small heat affected zone

Even more precise finish than flame cutting and plasma cutting

Cutting sharp corners – if we use regular laser beam, we will burn the material 🡺 we need a pulsed beam (which is not continuous) 🡺 it does not overheat the material

Laser source is outside the machine 🡺 we don’t move heavy machinery – we use flying optics (mirrors)
the light we are using is reflected on the sheet 🡺 reflection and conduction 🡺 problem

We want the highest energy density at 1/3 of the height of our sheet

Whatever you do in laser cutting, any cut part will have a typical X shape that will not be cut away 🡺 you will never have parallel cuts 🡺 the thicker the material, the bigger the X shape

Limitation of laser cutting – below the focus point of the non-parallel beam, the energy density will decrease 🡺 at some point the laser light will be so wide that it won’t cut anymore

Other laser processes
Laser drilling
Laser cutting - up to 30 mm thickness
Laser engraving
Laser welding
Laser soldering
Laser bending
Wire-EDM – we have a wire that goes comes close to the wokpiece but never touches it 
We use a high current 🡺 there is a high voltage difference between the wire and the wokpiece 🡺 when the wire comes close to the wokpiece, then at some point the room between the wire and the wokpiece which is filled by a dielectric liquid (not a conductor) cannot withstand the current anymore 🡺 there will be a flash of lightning between the wire and the product 🡺 small part of the product will be cut out 🡺 takes a long time 

Geometry - defined by how we move the wire
Accurate and smooth surface finish
Subtle process

Electrode – wire with diameter 0.05 to 0.5 mm
The wire gets damaged by every flashed, thus the machine adds new wire continuously
Thousands of flashes happen per second


Waterjet cutting – the pressure of the water is so high it can cut the material, water comes out of a jet (radius <1mm) [can cut paper without making it wet]
High pressure - ~400 Mpa (can cut paper without wetting it)

Can cut textile, plastics

Abrasive cutting – adding small particles of sand to cut glass or metal
Oxidation – can happen because we have water, oxygen and metal 🡺 rust

Not a very subtle process 

Lecture 5: Machining
Machining – removal of the material or modification of surfaces without changing the structure of the material
We remove the material by producing chips, using various tools
Allows us to make complex shape
Tool doesn’t have information about product geometry
Good dimensional accuracy – depends on the knowledge of the tool operator which translates from the tool to the material
Good surface finish
Advantage – we use generic tools, not special machines 🡺 we don’t have to order machines, dies 🡺 no long initial period
Properties
cutting
Rake angle – α
Clearance angle
 shearing
Shearing angle – Φ
Depth - t0
Important: Rake angle (α) + Tool angle (β) + Relief angle (γ) = 90°
	  here the α is negative

Exam question: describe + explain the elementary cutting 
The bigger the rake angle (the flatter the tool lies), the lower the forces will be to cut the material

We want to make the tool angle as big as possible because the bigger the tool angle, the stronger the tool tip will be 🡺 tool will last longer

Relief angle is needed because the workpiece locally heats up where the tool makes contact 🡺 its volume expands 🡺 it needs space to expand 





	
Operating factors
Cutting
Depth – pushing the tool into the workpiece
Speed – the rotation of the workpiece, number of revolutions per minute, the diameter
Feed – the amount of time it takes to cut away the material by moving the tool along the length of the workpiece (measured in movements of the tool per rotation of the workpiece) 
Fluids
Types
Oils - low speed, low temparature
Emulsions - high speed, high temperature
water and oil
Semi-syntetics 
little oil in water
Synthetics 
chemicals and additives in water

When to use?
can be added for cooling or changing the forces a bit
reduce friction and wear
flush chips away
protect machine from corrosion

When not to use?
Avoid fluids because they can be a chemical waste

Tool angle – adjusting it can change the forces
Chip type
Temperature rise – workpiece is relatively cool because most heat is removed by the chip
Tool wear – one of the bigger problems in machining
Thermal plastic deformation
Flank wear
Crater wear
Nose wear

Forces
Cutting force depends on:
Work piece material – the stronger the material, the higher the cutting forces
Wedge angle – the bigger the rake angle (the flatter the tool), the lower the forces
Feed – the lower the feed, the lower the cutting forces (potentially higher quality surface finish)
Depth – the lower the depth, the lower the cutting forces (potentially higher quality surface finish)

Cutting power
Power = Fc * vc

Tool characteristics
Hot hardness
Toughness and impact strength
Thermal shock resistance
Wear resistance
Chemical stability and inertness

Tool speed
Tool Steel - vc < 0,2 m/s
High Speed Steels (HSS) - vc = 0,3 up to 0,5 m/s
Hard metal (HM) (uncoated carbides) - vc = 1,5 up to 3,0 m/s
Coated carbides - vc < 6,0 m/s
Ceramics - vc < 20 m/s
Cubic Boron Nitride (CBN)
Diamond

Tool life
Taylor tool-life equation: vc * Tn = C
vc = cutting speed
the higher the cutting speed, the quicker the process will go
the higher the cutting speed, the higher the tool wear
T = tool life
n = heart of the tool (tool material)

C = constant
we can’t look it up or calculate it
value of C increases with the decrease in depth of the cut
value of C increases with the decrease in feed
value of C is affected by all factors around the machine – temperature, fluids, etc.

example to find C:
A lathe (turning machine) manufactures 12 products per hour
Tool:
Hard metal: n = 0.2
Tool life: T = 15 minutes
Speed: vc = 5,0 m/s
Tool change time:	5 minutes per change
A manager comes in and wants to increase production by increasing the cutting speed to vc = 5,4 m/s
What is the new amount of products that are produced every hour? (rounding off is allowed)

Answer:
Cutting length: 5,0 [m/s] x 60 [sec] x (3 [tools] x 15 [minutes]) = 13.500 m
For 12 products: = 1,125 m/product
Using Taylor:
5,0 x 150,2 = C = 5,4 x T0,2    🡪     T = 10 minutes
Cutting length: 5,4 [m/s] x 60 [sec] x (4 [tools] x 10 [minutes]) = 12,960 m
New situation: 12,960/1,125 = 11.52 products/hour

Warning (might be in the exam):  When we look at the graph, we can’t find a break-even point (point where curves cross) and say it is optimal. We can only add up the costs to meet the minimum point of the condition (the dotted line), which will give us the optimum.


ISO classification
For example, HC-P 15
HC = coated material, 
P = Work piece material
15 = mechanical load

Machining processes

Workpiece rotates
Turning 
Cutting off
Hole making

Wokpiece slides
Scraping

Tool rotates
Stab milling
End milling

Tool slides
Scraping
Broaching

Turning – the workpiece rotates, the tool is stationary 
Product geometry – rotational symmetry (straight, conical, curved/grooved)
Product materials – all kinds
Advantage – use of generic rules 🡺 no lead time 
Disadvantage – tool has to move along the whole geometry of the product 🡺 each product takes the same time to make 🡺 low production rate
Operations - cutting speed in the middle of the diameter of the material is 0, thus tool can’t cut it, the material just chips away

Parametres
Cutting speed 	vc = π * D * n
Cutting depth	d
Feed		f
Rotation		n 

Chip control – chips might be too big 🡺 if machine is controlled by a computer, computer doesn’t know how big the chips are 🡺 tool might get damaged
Solutions - chip breaker, rake

Surface finish
Explicit factors – feed rate (f), tool geometry (R)
Noise factors – build up edge, vibrations 



Tools


Product specific information
Contour turning
Copy turning

Boring – making a hole larger with as little force as possible (different from drilling, has high accuracy)

Drilling – making screw thread inside a hole 
Tool geometry – immediately determines the shape of the hole
Maximum hole depth is related to the tool diameter
Feed force – pushed against the material
Torque – the rotational force between the drill string and the formation
Driving power
Tool wear – determined by the remaining useful length of the drill after sharpening
Rake angle – varies over diameter 
Positive – outside
Negative – inside 
Important: when making holes in a thin sheet of metal, the metal has to be stabilized, else it will start spinning around the drill in the air
Chip angle – varies over the diameter of the drill 🡺 positive (outside diameter) and negative (near the core) chip angles

Milling 
Tool
moves with respect to the workpiece
rotating tool
feed is perpendicular to the axis of rotation
tool is like a flex, flat to the surface of the material

Parameters – rotational speed, feed, depth of cut

Two strategies - workpiece is pressed against the table, avoid high surface hardness, immediate cutting, not grating, high(er) surface quality, requires slackless machine
Conventional milling (up milling) – firmly starts below the hard layer 🡺 much easier
Chip load on teeth slowly increases from 0 to max
Chip chatter (vibrations) possible
Chip burr (formed on unfinished surface) is removed automatically with consecutive rotations
Lower quality surface finish
Contamination or scale (oxide layer) on the surface does not affect tool life
Workpiece is pulled upward 🡺 we need proper clamping
Climb milling (down milling) – firmly hits the thickest part of the material
Chip load on teeth goes directly to max, then decreases from max to 0
Chip burr (formed on finished surface) is not removed automatically
Much more favorable way of milling
Rotation of the cutter pushes workpiece downward 🡺 holds it in place
Method is not suitable for workpieces with surface scale (hot worked/forged/cast metal) because the scale is hard and abrasive 🡺 excessive wear and damage of the cutter teeth 🡺 shorter tool life

Grinding 
Tool – disc with a large number of undefined cutting edges
Hardness of the disc – determined by the strength of the binding between the grains (not by the strength of the grains themselves), because if binding is weak 🡺 grains break away more easy from the grinding disk
Dull grains burst out

Structure of the disc - pore size
Contact length
Determines the optimal tool type
Plane grinding lc
Large contact length for internal cylindrical grinding
Determines productivity Q’
Grinding energy Pc 

Grains have unpredictable shapes
Grain size
Influences surface quality
Influences removal speed

Speed of cutting – not very fast because grains take away very small chips of material
At high revolution speeds, grinding disks can’t be cooled effectively – the cooling fluids can hardly reach the disk because of centrifugal forces
Abrasive grains have a strong negative chip angle - Al2O3, SiC, CBN, diamant
Specific cutting force - much higher compared to milling
Small machined volume per minute
Expensive process
Application
High accuracy
high surface finish
hard material
low processing forces
thin layers can be removed	
Grinding temperature
Heat drain: 70% in the workpiece
High temperatures
Change of material structure
Surface cracks
Geometric inaccuracies
Cooling
Water-oil emulsion

Centerless grinding
Workpiece is clamped between grinding wheel and regulating wheel
Through-feed grinding

Physical/electrical and chemical machining
Electrical-discharge machining (EDM) – we provide a current and we create electrical discharge with it, which will melt/evaporate extremely small particles; the whole process happens under a fluid (dielectric fluid); because we are taking away so little material, we need as much electrical discharge as possible 🡺 pulse generator
Electrode - 100 – 300 V; 5 – 200 A
Dielectric fluid - non-conductive fluid, tries to separate the electrode from the workpiece
If we use conductive fluid 🡺 short circuit 🡺 no sparks
Pulse generator - 50 – 1000 kHz
Rinsing system – takes away the excess material from the fluid because if too much of the material remains in it, the fluid will become conductive
Slow but precise process
Gap width control system – needed for the careful positioning of the tool
gap: 5 – 100 μm
applications
Materials with high hardness – the tool and material never touch 🡺 we don’t need any force 🡺 EDM has the least requirements as concerns strengths and stiffness
Complex shapes – as long as we can make the shape of the electrode
No or low forces – the tool and material never touch
Good geometric and surface quality
Low removal rate
Electrode wear – the spark not only hits the workpiece but also damages the electrode
Suitable for tool production (e.g. molds) 




Electro-chemical machining (ECM) – an electrode is lowered into the workpiece to shape it, in this case the fluid is a salty saline electrolyte (conducts electricity) 🡺 material is dissolved (electrolysis)
Relatively high removal rate
Proportional to strength of current (amperage)
5 – 20 V
>10.000 A
Limited accuracy – the faster we remove material, the less accurate the process will be
No tool wear – we don’t have sparks to erode the tool
If we use AC (alternating current) 🡺 ionization at the electrode 🡺 it wears away, also material removal happens 50% of the time
Environmental hazard – because of the chemicals we are using
Corrosion problems – might be caused on the material by the saline liquid

Electrolyte

Chemical machining (Etching) – we have a very aggressive chemical fluid (and no electrode) which eats away the material
Application
‘Chemical attack’; no current
Chemically aggressive fluids
No inversion of geometry
No electrode
No machining forces
Machining of extremely thin workpieces possible
No burrs

EDM vs ECM
ECM - low voltage, high current 
EDM - high voltage, lower current


Lecture 6: Joining
  Joining – processes that bring all components together into one assembly
if one component is not there at the right time in the right amount, the whole production stops
magnetism – important issue because components stick together during the assembly
some components of an assembly are an assembly of smaller components themselves 🡺 in some cases there is a hierarchy of assembly

Detachability 
Fully detachable – you can take the assembly apart without damaging the components
Partially detachable – plastic components that click together (snap fit)
Not detachable – the connection between parts cannot be undone (e.g. - glue)

Joining principle
Geometry – do we connect the geometry?
Force – do we force components so they fit?
Material – do we connect components by melting them?

Geometry
Point – components meet at a point
Line - components meet at a line
Face - components meet at a face


Joining processes


Fusion welding – melting materials to join them with materials of similar compositions and melting points
Oxyfuel-gas welding – we use gas, which comes out of the torch and we set fire to the gas

Advantage – we don’t need electricity 🡺 thermochemical welding process
Disadvantage – not a subtle process because we are burning gas 🡺 flame cannot be small 🡺 doesn’t work with thin materials



(Gas-)Tungsten arc welding (TIG) – we use inert gas to push oxygen away (so tungsten doesn’t wear away) and electricity to make a voltage difference between the material and the electrode (made out of tungsten, doesn’t wear away) 🡺 the electricity in the arc will make the arc more precise


Shielded Metal-arc Welding (old-fashioned method) – the electrode is made by a metal used to conduct the electricity and has coating that contains the small bubbles of gas that we need, electrode burns away 🡺 releases the gas
Advantage – we don’t have a separate cable with gases and fillers 🡺 one of the hands of the operator is free 🡺 they can position the workpiece better 🡺 higher accuracy

Disadvantage – operator has to control the distance between the electrode and the workpiece quite accurately 🡺 process is difficult to be done by robot because it can’t accurately predict when the electrode is wearing away

Submerged-arc Welding

Gas Metal-arc Welding – electrode is made from a material that wears away 🡺 we put the material on a wire and we push it so it becomes the new electrode 🡺 electrode itself is a filler material, that conducts electricity 🡺 we provide new electrode material non-stop 🡺 distance between torch and base material is constant

Active gas 🡪 MAG (Metal Active Gas), CO2
Electrode is metal, not tungsten
active gas – contains oxygen 🡺 it starts to burn when it comes in contact with the base material
oxygen 🡺 more heat 🡺 material burns away quicker
oxygen 🡺 risk of rust, corrosion, oxygenation in the base material

Inert gas 🡪 MIG (Metal Inert Gas)

Flux-cored Arc Welding
Electrogas Welding
Electroslag Welding
Beam Welding 
Electron-beam 
Laser-beam

Plasma-arc welding – similar to Tungsten-arc welding as the arc is formed between a pointed tungsten electrode and the workpiece. However, by positioning the electrode within the body of the torch, the plasma arc can be separated from the shielding gas envelope

Other welding (Solid state welding) – we locally heat the material up so that it would locally melt under the influence of mechanical, electrical, or thermal energy
Cold Welding

Ultrasonic Welding - high-frequency ultrasonic acoustic vibrations are locally applied to workpieces being held together under pressure 

Friction Welding – we connect two components by keeping one of them steady and rotating the other one, while pushing them together
Strong connection – when rotating, all grains will be pushed to the outside 🡺 clean weld, no air traps
We have to rotate on of the components 🡺 difficult with bigger parts


Explosion Welding – materials explode and the force of the explosion welds them together




Resistance welding – we don’t need gas, we have two electrodes and a current difference between them, we put the two materials between them, we add electricity and force push them together 🡺 there is resistance between the two materials 🡺 they locally heat up 🡺 the temperature welds them together

Surface of the material is not perfectly smooth 🡺 
The smoother the surface 🡺 the lower the resistance

Non-detachable process – once done, the process can’t be undone

Fast process

We know where we are going to weld between the two electrodes 🡺 we can use automation
Types of resistance welding
Projection welding – connect two materials in multiple spots in one go


Flash Welding – pushing together two materials that are not rotating, we add a lot of force until they are welded together




Stud Welding

Spot and seam welding 
Seam welding – using electrode wheels to “roll” the products together and thus weld them
continuous AC (alternating current)
electrically conducting rollers
Spot welding
alternating current - 3000 to 40 000 A
copper electrodes – sufficiently electricity conducting
workpieces should also be condustive
uses a single point electrode
high-frequency resistance welding
high-frequency current - up to 450 kHz

Fastening and bonding
Riveting
The rivets that connect the components have to be made of a softer material than the materials
Softer rivets allow a bit of movement, unlike welding which makes the joint pretty sturdy
permanent or semi-permanent mechanical joining

Joining faces (gluing) - two components and an intermediate between them (is usually less strong than the material of the components), the bigger the surface area of the intermediate (glue) 🡺 the stronger the connection, intermediate or relatively slow strength needed
Adhesive – the strength of the connection between the intermediate and workpiece
Glue – organic material, resin
To increase the quality of the adhesion 🡺 clean surface area so that we don’t glue dust instead of the components
Materials
Porous materials – bonding
Plastics – softening of the surface layer
Metals – van der Waals-forces

Cohesive – the strength of the intermediate itself
Thin layer of glue
if the materials aren’t sticking together, adding more glue won’t help, because we are just adding more of the same weak stuff 🡺 increase surface area to increase strength
Brazing, soldering – we heat up the components and add a softer material in order to have cohesion between all the materials, the softer material will melt and perfectly fill the space between the components (capillary effect) 












Seaming – the two components are rolled together and thus locked
		      

Crimping -  joining (ductile) components by deforming one or both of them to hold the other




Lecture 7: Plastic processing
Most processes for plastics are Net-shape or near-net-shape – we aim to make the part in one process and immediately finish the process
Every extra movement (like going back to make holes) in mass produced products is very expensive

Plastics = polymer + additives
Polymer - extremely large molecule (from Greek: Poly=many, meras=parts), consisting of a repetitive pattern of an unspecified number of monomers

polymers are expensive on their own 🡺 we want to add additives
Plasticisers – makes plastic ore flexibility and softness
Anti-scission - protects against ultra-violet radiation and oxygen (because plastic disintegrates under UV light)
Fillers – reduces cost by adding sawdust, sand, chalk, …, asbestos
Colorants - organic (dyes) or inorganic (pigments)
Flame retardants - chlorine, bromine, ..., phosphorus
Lubricants - reduce friction/sticking during processing

Types of plastics
TP - Thermoplastic
TS - Thermosetting
E – Elastomer

Composites - a combination of chemically distinct and insoluble phases with a recognizable interface

Fibers
Glass fiber
Aramids (twaron, kevlar)
Carbon fiber

Extrusion - shaping viscous plastics by pushing them through mold/die 
Can only be done with thermoplastics and elastomers
Not with thermosetting plastics because we need viscous material, so we need to heat it up, which is impossible with thermosetting plastics

Injection molding – filling a mold with high pressure (70-200Mpa) so the viscous material can fill up every part of the mold
Used in mass production
High accuracy
Very good surface finish
If we want shiny surface, die also has to be shiny; every defect on the die will be replicated on the product; we also want the die to be as cold as possible to cool down the products 🡺 cooling channels 🡺 die has to be a very good quality and to be complex 🡺 dies are extremely expensive 
Complex products
Make materials into one go – biggest advantage
Material
thermoplastics, elastomers – heat up the material until it becomes viscous and then push it into the mold
thermosetting plastics – put all monomers/polymers into the die, chemical reaction that creates the thermosetting plastic happens inside the die 🡺 thermosetting plastic comes into existence inside the mold and can never be heated up again

Rotational molding – we pour the material into the mold and start rotating the mold 🡺 plastic will stick to the walls 
Large product geometries possible
Considerable variance in wall thickness 🡺 lower quality surface finish
Slow process 🡺 small batches 
We rotate die as randomly as possible
Dies are fairly cheap
We can make hollow products 🡺 they are fairly light
We have to heat up the material so it becomes viscous and sticks to the wall 🡺 we use thermoplastics, not thermosetting plastics

 Extrusion blow-molding – we extrude a tube, the die closes around it, a blow pin blows air from the other side and tube assumes the shape of the die 
By definition, the extruder will have cross sections 🡺 so will the extruded tube
The material hardly deforms when the diameter is small
the bigger the diameter of the mold, the thinner the walls will be 
example – water bottles have thin walls except for their necks and bottoms which are pretty sturdy
fairly cheap process 🡺 mass production
material – we heat up the plastic 🡺 thermoplastic 
the extruded material is viscous 🡺 without immediate cooling, increasing the air pressure would result into “blowing holes”




Injection blow-molding – we make a very small shape of what we actually want in an injection molding machine, then we put it into a bigger mold, we blow it up until the material touches the die
Injection 🡺 We can make different wall thicknesses 🡺 there will be less difference between wall thickness and bottoms and necks of bottles
Process is more precise 🡺 more expensive 


Calendaring – process of taking material and pressing it together with rolls 🡺 creating foil/film/sheet 
Product goes between rolls

Unpredictable process 🡺 once we set up the machine we want it to run for as long as possible


Thermoforming – plastic sheet is heated, we push the die up 🡺 it creates vacuum 🡺 the sheet assumes the form of the die, then sheet is trimmed to create a usable product (example – creating a cutlery tray ) 
Base material: foil or sheet
Thermoplastic 

Heating
Oven
Infrared

Vacuum forming
Combination with pressure or pressing

Relatively cheap molds
Low forces
Varying wall thicknesses

Lecture 8: Additive processes
  Additive manufacturing/rapid manufacturing - technologies that grow three-dimensional objects one (thin) layer at a time
From ‘exotic’ processes for making geometric prototypes
Via ‘simple’ processes for product with limited mechanical loads
To ‘fully functional’ products

Principles
Fast start of production – we only need the material and the machine
Trying to avoid complicated process planning
Processes that gradually build the product
Relatively low production speeds 🡺 small batches

Highly specialized products – not used for mass production (e.g. – making a part of a person’s skull)
User/consumer oriented

Surface quality
Good on flat surfaces
On steep surfaces (e.g. - the top of the head of a statue) you can see the layers of material (staircase effect)

No advantage to scale – each product takes exactly the same time to make

Complexity is ‘free’ - the technology can manufacture complex items that are very difficult or even impossible to make with extractive technologies

Speed up design - owing to the ability to manufacture “one-offs,” engineers can try many different designs and test them before an item is released to manufacturing

Customization - unique items can be made to exact fits, be it custom clothes or custom replacement body parts

Efficiency - almost zero waste

Weight reduction - only strength where required
Topology optimisation

Stereolithography – shining a light on the liquid locally will turn it into a solid material (Local solidification)
Density of the solid material should me sort of equal to the density of the liquid


Fused deposition modelling – we feed a thermoplastic wire through a heated built head 🡺 it becomes viscous 🡺 we draw with the viscous material
 

3D printing


Selective laser sintering – we add some powder, we bake the material together, we repeat the process until we have the result, then we remove the excess powder 


Laminated object manufacturing – we have one layer of material; we glue it locally to the next layer
very fast process
inexpensive 





Appendix: Surface finish roughness of various manufacturing processes


"""


# get data
from transformers import AutoTokenizer

chunk_size = 1024
chunker = semchunk.chunkerify("gpt-4", chunk_size)

chunks = chunker(text)

data_rows = [{"content":c} for c in chunks]


# data_rows = [
#     {"content": "Adidas is a globally recognized brand known for its innovation and craftsmanship in the footwear industry. Its shoes are designed with a blend of style, performance, and comfort, catering to a wide range of athletes and casual wearers. Whether it's for running, basketball, soccer, or fashion, Adidas offers a diverse array of products that appeal to different lifestyles and preferences."},
#     {"content": "One of the key innovations from Adidas is the use of Boost technology. Introduced in 2013, Boost revolutionized the sneaker industry by providing unmatched cushioning and energy return. Made from thermoplastic polyurethane (TPU), these shoes offer a bouncy, responsive feel, making them ideal for both athletes and casual runners who need extra comfort."},
#     {"content": "Adidas also stands out for its collaborations with high-profile designers, athletes, and celebrities. Partnerships with Kanye West for the Yeezy line, Pharrell Williams, and Stella McCartney have further cemented Adidas as a brand that blurs the lines between sportswear and high fashion. These collaborations result in limited-edition releases that generate significant buzz."},
#     {"content": "The Ultraboost series is one of Adidas' most popular running shoes. Known for its superior comfort and responsiveness, Ultraboost shoes are highly favored by runners. The design, featuring a sock-like Primeknit upper and a Boost midsole, provides a snug fit and maximum energy return, making long-distance running more enjoyable."},
#     {"content": "Sustainability is a growing focus for Adidas, and the brand has made strides in eco-friendly shoe production. Initiatives like the Parley collaboration involve using recycled ocean plastic in their shoes, helping reduce environmental impact. Adidas also aims to use recycled materials across the majority of its products by 2025, reflecting its commitment to sustainability."},
#     {"content": "Adidas Originals is another iconic line, representing the brand’s heritage and timeless appeal. Shoes like the Superstar, Stan Smith, and Gazelle are part of this collection, recognized for their classic designs that transcend generations. The Adidas Superstar, in particular, has been a staple in streetwear culture since the 1970s."},
#     {"content": "Soccer is one of Adidas' core sports, and the brand has produced some of the most iconic soccer boots, such as the Predator and Copa Mundial. These shoes are known for their superior control, traction, and durability, catering to the needs of both professional players and weekend warriors. The Predator, in particular, is celebrated for its precision and power."},
#     {"content": "Adidas' basketball shoes have also made waves in the sports world, with models like the Harden Vol. series designed for NBA star James Harden. These shoes are engineered to support explosive movements on the court, featuring cushioning technologies and sturdy builds to handle the intensity of the game."},
#     {"content": "The NMD line from Adidas is a perfect example of how the brand fuses innovation with street style. NMD shoes incorporate Boost technology and a minimalist, futuristic design, making them popular among sneaker enthusiasts. They’re highly versatile and can be worn for casual outings or athleisure activities."},
#     {"content": "Adidas places a strong emphasis on innovation, which is reflected in technologies like Primeknit. Primeknit is a lightweight, seamless upper that offers flexibility and breathability. This material molds to the shape of the foot, providing a custom fit that enhances performance, particularly in running and training shoes."},
#     {"content": "The Yeezy line, a collaboration between Adidas and Kanye West, has become one of the most sought-after sneaker collections in the world. Known for their unique designs and limited availability, Yeezy sneakers have garnered a cult following. The Yeezy Boost 350 and 700 models, in particular, are praised for their comfort and bold aesthetics."},
#     {"content": "Adidas’ commitment to performance is also evident in its running shoes, like the Adizero series. These shoes are lightweight and designed for speed, often used by professional athletes during marathons and track events. The Adizero Adios Pro, for example, is known for its carbon-infused energy rods that provide propulsion and a fast, responsive ride."},
#     {"content": "In recent years, Adidas has embraced the growing athleisure trend, creating shoes that are stylish enough for everyday wear yet functional enough for workouts. Models like the Adidas Nite Jogger and ZX series exemplify this trend, with retro-inspired designs that appeal to both sneakerheads and casual wearers alike."},
#     {"content": "Adidas shoes are also known for their durability, with models like the Terrex line designed for outdoor adventures. These shoes are built to withstand harsh conditions, offering features like Gore-Tex waterproofing and Continental rubber outsoles for superior grip on rugged terrains. They’re ideal for hiking, trail running, and other outdoor activities."},
#     {"content": "Comfort is a priority for Adidas, and the brand has continuously improved its footwear to meet the needs of its customers. The Cloudfoam technology, for instance, provides soft cushioning that is perfect for all-day wear, whether for casual strolls or standing for long periods. It’s a popular choice for those seeking lightweight, comfortable footwear."},
#     {"content": "Adidas' focus on customization is another key feature that sets it apart. The brand allows customers to personalize their shoes with the mi Adidas platform, where they can choose colors, materials, and even add custom text to make their sneakers truly unique. This level of personalization appeals to individuals looking for one-of-a-kind footwear."},
#     {"content": "Adidas shoes are also favored by athletes for their performance-enhancing features. The brand's running shoes, in particular, are known for their lightweight construction and supportive midsoles, which help reduce fatigue during long runs. This focus on comfort and performance makes Adidas a top choice for both professional and amateur runners."},
#     {"content": "The Continental rubber outsole is another standout feature in many Adidas shoes, offering excellent traction on various surfaces. This technology is borrowed from the tire industry and ensures that athletes can perform at their best, whether they’re running on wet pavement, trails, or a basketball court."},
#     {"content": "Adidas’ influence extends beyond sports, as the brand has become a cultural icon. Whether it's through collaborations with artists or its strong presence in streetwear fashion, Adidas shoes are often seen as a status symbol. The brand’s ability to stay relevant in both performance sports and fashion is a testament to its versatility and innovation."},
#     {"content": "Innovation doesn’t stop with materials and design for Adidas; the brand also invests in technologies like 3D printing. Adidas' 4D shoes, created using a 3D-printed midsole, offer a unique combination of support and cushioning. This cutting-edge approach ensures a more customized fit based on individual movement patterns."},
#     {"content": "In conclusion, Adidas shoes represent the perfect blend of style, innovation, and performance. Whether you’re an athlete looking for high-performance footwear or someone seeking a stylish yet comfortable pair for daily wear, Adidas offers a wide range of options to meet your needs. Its continued focus on innovation, sustainability, and cultural relevance keeps it at the forefront of the footwear industry."}
# ]

# for s in data_rows:
#     print(s)
#     print("-" * 70)


with collection.batch.dynamic() as batch:
    for data_row in data_rows:
        batch.add_object(
            properties=data_row,
        )

connection.close()
print("done")